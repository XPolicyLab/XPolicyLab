import re
from typing import Optional
from dataclasses import dataclass


# Takes in the full A11Y tree, filters out nodes that are visible to the user and contain
# bounding boxes, then returns this information in the form of bounding boxes (left, top, right, bottom),
# the center coordinates and sizes of each box, and extra meta info for visualization purposes.
# NOTE: if boundsInScreen does not contain all 4 bounds, will default to 0 for left,top and w,h for right,bottom.
def extract_bbs_from_a11y(a11y_data, dim):
    filtered_nodes = []
    for idx, window in enumerate(a11y_data["windows"]):
        for node in window.get("tree", {}).get("nodes", []):
            bounds = node.get("boundsInScreen", {})

            has_left = "left" in bounds
            has_top = "top" in bounds
            has_right = "right" in bounds
            has_bottom = "bottom" in bounds
            if not (has_left or has_top or has_right or has_bottom):
                continue

            if not node.get("isVisibleToUser", False):
                continue
            filtered_nodes.append(node)

    bbs = []
    bb_centers = []
    bb_sizes = []
    extra_meta_strings = []
    for node in filtered_nodes:
        bounds = node["boundsInScreen"]

        left = bounds.get("left", 0)
        top = bounds.get("top", 0)
        right = bounds.get("right", int(dim[0]))
        bottom = bounds.get("bottom", int(dim[1]))

        box_area = (right - left) * (bottom - top)

        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)

        size_x = (right - left)
        size_y = (bottom - top)

        bb_centers.append((center_x, center_y))
        bb_sizes.append((size_x, size_y))
        bbs.append([left, top, right, bottom])

        extra_meta_str = 'Box: ' + str([left, top, right, bottom]) + ' Center:' + str([center_x, center_y]) + ' Size:' + str([size_x, size_y]) + ' Metadata: ' + node.get('text', '') + ' ' + node.get('viewIdResourceName', '')
        extra_meta_strings.append(extra_meta_str)
    return bbs, bb_centers, bb_sizes, extra_meta_strings


# Given a string containing the reduced A11Y tree, searches for some {search_text},
# and if found, will return the bounding box of the element corresponding to it.
# This is a very specific function meant to deal with click(coordinate of $app_name) == open $app_name.
def extract_app_bb(input_string, search_text="text=Note"):
    input_string = input_string.lower()
    if search_text in input_string:
        pattern = re.compile(rf"center=\[(\d+),(\d+)\],\s*size=\[(\d+),(\d+)\],\s*{search_text}")
        match = pattern.search(input_string)
        if match:
            center_x, center_y = int(match.group(1)), int(match.group(2))
            width, height = int(match.group(3)), int(match.group(4))

            left = center_x - width // 2
            top = center_y - height // 2
            right = center_x + width // 2
            bottom = center_y + height // 2
            return (left, top, right, bottom)
    return ''


# Given the target coordinates of a click or longpress action and a list of
# bounding boxes and their center coordinates and size for the current screen,
# look for centers that are within +/- 1 of the target coordinates. If unsuccessful,
# define the target bounding box as +/- 10 of the target coordinates.
# Strategy will either be center or smallest.
def find_gt_box(gt_coords, bb_centers, bb_sizes, bbs, strategy='center'):
    gt_box = None

    # If the center of an element's bounding box == +/- 1 of target coordinates...
    if strategy == 'center':
        smallest = None
        x, y = int(gt_coords[0]), int(gt_coords[1])
        for i,bb_center in enumerate(bb_centers):
            if abs(bb_center[0] - x) <= 1 and abs(bb_center[1] - y) <= 1:
                bb_size = bb_sizes[i]
                if smallest is None:
                    smallest = bb_size
                    gt_box = bbs[i]
                elif bb_size[0] < smallest[0] and bb_size[1] < smallest[1]:
                    smallest = bb_size
                    gt_box = bbs[i]
                else:
                    continue

    # The smallest bounding box that encloses the target coordinates...
    elif strategy == 'smallest':
        enclosing_bbs = []
        for i,bb in enumerate(bbs):
            if within_bounding_box(gt_coords, bb):
                enclosing_bbs.append({'bbox': bb, 'size': bb_sizes[i]})
        if len(enclosing_bbs) > 0:
            enclosing_bbs.sort(key=lambda item: item['size'])
            gt_box = enclosing_bbs[0]['bbox']

    gt_box = gt_box if gt_box != None else (gt_coords[0] - 10, gt_coords[1] - 10, gt_coords[0] + 10, gt_coords[1] + 10)
    return gt_box


# Simple function that returns True if the provided coordinates are enclosed by
# the provided box, and False otherwise.
def within_bounding_box(coords, box):
    x, y = int(coords[0]), int(coords[1])
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class Element:
    class_: Optional[str] = None
    package: Optional[str] = None
    resource: Optional[str] = None
    bbox: Optional[dict] = None
    text: Optional[str] = None
    properties: Optional[list] = None

    def __str__(self):
        l, t, b, r = self.bbox.get("left", 0), self.bbox.get("top", 0), self.bbox.get("bottom", 0), self.bbox.get("right", 0)
        box_w = int(r) - int(l)
        box_h = (b) - (t)
        c_x = int((l) + (box_w) / 2)
        c_y = int((t) + (box_h) / 2)

        a11y = f"center=[{c_x},{c_y}], size=[{box_w},{box_h}]"

        if self.text not in ['', None]:
            a11y += f", text={self.text}"
        elif self.resource not in ['', None]:
            a11y += f", resource={self.resource}"

        if self.class_ != None:
            if 'Switch' in self.class_:
                a11y += ", text=Switch"
            elif 'Edit' in self.class_:
                a11y += f", EditableText"

        if len (self.properties) > 0:
            a11y += ", " + str(self.properties)

        return a11y


def reduce_a11y_tree(a11y_tree):
    def extract_node_info(node):
        properties = []
        for prop in ["isCheckable", "isChecked", "isClickable", "isClicked", "isLongClickable", "isScrollable", "isSelected"]:
            if node.get(prop, False) != False:
                properties.append(prop)

        bbox = node.get("boundsInScreen", {})

        pattern = r'com.*?id/'
        resource = node.get("viewIdResourceName", "")
        resource = re.sub(pattern, '', resource)

        if node.get("className") != None:
            cls = node.get("className").replace("android.widget.", "")
        else:
            cls = ""

        node_info = {
            "package": node.get("packageName"),
            "resource": resource,
            "bbox": bbox,
            "class_": cls,
            "properties": properties
        }
        if node.get("text") not in ['', None]:
            node_info['text'] = node.get("text")
        return node_info

    def traverse_nodes(nodes):
        reduced_nodes = []
        for node in nodes:
            if node.get("isVisibleToUser"):
                if node.get("isClickable") or node.get("isCheckable") or node.get("text") != "" or node.get("viewIdResourceName") != "":
                    node_info = extract_node_info(node)
                    new_node_info = {}
                    for k,v in node_info.items():
                        if not v in [False, '', None, 'None']:
                            new_node_info[k] = v
                    reduced_nodes.append(new_node_info)
        return reduced_nodes

    reduced_tree = []
    for window in a11y_tree.get("windows", []):
        if "tree" in window:
            nodes = window["tree"].get("nodes", [])
            nodes = remove_duplicates(nodes)
            reduced_tree.extend(traverse_nodes(nodes))

    package_to_els = {}
    for node in reduced_tree:
        el = Element(**node)
        if el.package not in package_to_els:
            package_to_els[el.package] = []
        package_to_els[el.package].append(el)

    formatted = []
    for package, els in package_to_els.items():
        formatted.append(package)
        for el in els:
            formatted.append("\t" + str(el))
        formatted.append("")

    formatted = "\n".join(formatted)
    return formatted


def remove_duplicates(a11y_tree):
    seen = set()
    unique_tree = []

    for element in a11y_tree:
        identifier = (element.get('bbox'), element.get('class'), element.get('resource'), element.get('text'))
        if identifier not in seen:
            seen.add(identifier)
            unique_tree.append(element)
    return unique_tree