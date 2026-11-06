"""Bind one installed NVIDIA Vulkan ICD and resolve renderer devices by UUID.

CUDA/NVML physical indices are not Vulkan indices. Probe the driver in a fresh
short-lived process before Kit/model initialization; never install/delete drivers.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def select_driver(environment, candidates=None):
    explicit = {environment[k] for k in ('VK_DRIVER_FILES', 'VK_ICD_FILENAMES') if environment.get(k)}
    if len(explicit) > 1:
        raise ValueError('Vulkan driver overrides disagree; select one NVIDIA ICD')
    if explicit:
        value = next(iter(explicit))
        if os.pathsep in value:
            raise ValueError('Only one Vulkan ICD may be selected')
        paths = [Path(value)]
    else:
        paths = list(candidates or (Path('/etc/vulkan/icd.d/nvidia_icd.json'),
                                   Path('/usr/share/vulkan/icd.d/nvidia_icd.json')))
    path = next((p for p in paths if p.is_absolute() and p.is_file()), None)
    if path is None:
        raise ValueError('Missing installed NVIDIA Vulkan ICD')
    content = path.read_bytes()
    data = json.loads(content)
    library = data.get('ICD', {}).get('library_path', '')
    if not isinstance(library, str) or 'nvidia' not in library.lower():
        raise ValueError('Expected the installed NVIDIA Vulkan driver')
    environment.update(VK_DRIVER_FILES=str(path), VK_ICD_FILENAMES=str(path))
    return dict(driver_file=str(path), driver_file_sha256=hashlib.sha256(content).hexdigest(),
                library=library)


def choose_renderer(devices, expected_uuid):
    identities = [row['uuid'].lower() for row in devices]
    if not devices or len(set(identities)) != len(identities):
        raise ValueError('Vulkan reports duplicate/empty GPU identities; do not start Kit')
    matches = [row for row in devices if row['uuid'].lower() == expected_uuid.lower()]
    if len(matches) != 1 or matches[0]['vendor_id'] != 0x10de:
        raise ValueError('Leased GPU UUID is not uniquely available through the selected NVIDIA ICD')
    # Kit's GPU Foundation table orders NVIDIA adapters by PCI address; the
    # driver's raw vkEnumeratePhysicalDevices order can differ from that table.
    addresses = [tuple(row.get('pci', [])) for row in devices]
    if (any(len(pci) != 4 or any(type(n) is not int or n < 0 for n in pci) for pci in addresses)
            or len(set(addresses)) != len(addresses)
            or any(row['vendor_id'] != 0x10de for row in devices)):
        raise ValueError('Renderer admission requires unique NVIDIA PCI identities')
    ordered = sorted(devices, key=lambda row: tuple(row['pci']))
    return ordered.index(matches[0])


def configure(environment, expected_uuid, *, probe=None):
    proof = select_driver(environment)
    if probe is None:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--enumerate']
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30, check=True)
        devices = json.loads(result.stdout)
    else:
        devices = probe(environment)
    index = choose_renderer(devices, expected_uuid)
    environment['RIEMANN_RENDERER_GPU_INDEX'] = str(index)
    return dict(proof, expected_uuid=expected_uuid, renderer_index=index, devices=devices)


def enumerate_devices():
    """Query Vulkan 1.1 deviceUUIDs without importing Kit, Torch or task code."""
    import ctypes as c
    import uuid

    class Application(c.Structure):
        _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('name', c.c_char_p),
                    ('version', c.c_uint32), ('engine', c.c_char_p), ('engineVersion', c.c_uint32),
                    ('apiVersion', c.c_uint32)]
    class Create(c.Structure):
        _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                    ('application', c.POINTER(Application)), ('layerCount', c.c_uint32),
                    ('layers', c.c_void_p), ('extensionCount', c.c_uint32), ('extensions', c.c_void_p)]
    class Identity(c.Structure):
        _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('deviceUUID', c.c_ubyte*16),
                    ('driverUUID', c.c_ubyte*16), ('luid', c.c_ubyte*8),
                    ('nodeMask', c.c_uint32), ('luidValid', c.c_uint32)]
    class Properties(c.Structure):
        # VkPhysicalDeviceProperties is a fixed Vulkan core output structure.
        # Use an oversized, eight-byte-aligned output buffer; only the vendor
        # header is read. Identity comes from the typed pNext structure above.
        _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('properties', c.c_uint64*512)]
    class PCI(c.Structure):
        _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('domain', c.c_uint32),
                    ('bus', c.c_uint32), ('device', c.c_uint32), ('function', c.c_uint32)]
    library = c.CDLL('libvulkan.so.1')
    library.vkCreateInstance.argtypes = [c.POINTER(Create), c.c_void_p, c.POINTER(c.c_void_p)]
    library.vkCreateInstance.restype = c.c_int32
    library.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.POINTER(c.c_void_p)]
    library.vkEnumeratePhysicalDevices.restype = c.c_int32
    library.vkGetPhysicalDeviceProperties2.argtypes = [c.c_void_p, c.POINTER(Properties)]
    library.vkGetPhysicalDeviceProperties2.restype = None
    library.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    library.vkDestroyInstance.restype = None
    app = Application(sType=0, name=b'Riemann-RSI-Preview-device-query', apiVersion=(1<<22)|(1<<12))
    create = Create(sType=1, application=c.pointer(app))
    instance = c.c_void_p()
    code = library.vkCreateInstance(c.byref(create), None, c.byref(instance))
    if code != 0:
        raise RuntimeError('Vulkan instance creation failed: ' + str(code))
    try:
        count = c.c_uint32()
        if library.vkEnumeratePhysicalDevices(instance, c.byref(count), None) != 0 or not 0 < count.value <= 64:
            raise RuntimeError('Invalid Vulkan device enumeration')
        devices = (c.c_void_p*count.value)()
        if library.vkEnumeratePhysicalDevices(instance, c.byref(count), devices) != 0:
            raise RuntimeError('Vulkan devices changed during enumeration')
        rows = []
        for index in range(count.value):
            pci = PCI(sType=1000212000)
            identifier = Identity(sType=1000071004, pNext=c.cast(c.pointer(pci), c.c_void_p))
            properties = Properties(sType=1000059001, pNext=c.cast(c.pointer(identifier), c.c_void_p))
            library.vkGetPhysicalDeviceProperties2(devices[index], c.byref(properties))
            header = c.cast(c.byref(properties, Properties.properties.offset), c.POINTER(c.c_uint32))
            rows.append(dict(index=index, uuid='GPU-'+str(uuid.UUID(bytes=bytes(identifier.deviceUUID))),
                             vendor_id=header[2], pci=[pci.domain, pci.bus, pci.device, pci.function]))
        return rows
    finally:
        library.vkDestroyInstance(instance, None)


if __name__ == '__main__':
    if sys.argv[1:] != ['--enumerate']:
        raise SystemExit('Use --enumerate for the read-only Vulkan device query')
    print(json.dumps(enumerate_devices()))
