#!/bin/bash

# CAN interface configuration script for Piper arms
# Automatically detects, renames, and configures CAN interfaces based on USB ports
# Prerequisites: ethtool, can-utils, gs_usb driver
# Usage: sudo bash can_config.sh [interface_name] [bitrate] [usb_address]

# Expected number of CAN modules
EXPECTED_CAN_COUNT=2

if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    # Default CAN interface name (can be set via command line argument)
    DEFAULT_CAN_NAME="${1:-can0}"

    # Default bitrate for single CAN module (can be set via command line argument)
    DEFAULT_BITRATE="${2:-1000000}"

    # USB hardware address (optional parameter)
    USB_ADDRESS="${3}"
fi

# Predefined USB ports, target interface names and bitrates (for multiple CAN modules)
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS
    USB_PORTS["1-13:1.0"]="can_left:1000000"
    USB_PORTS["1-12:1.0"]="can_right:1000000"

fi

# Get current number of CAN modules in the system
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Check if current CAN module count matches expected count
if [ "$CURRENT_CAN_COUNT" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "Error: Detected CAN modules ($CURRENT_CAN_COUNT) does not match expected count ($EXPECTED_CAN_COUNT)."
    exit 1
fi

# Load gs_usb module
sudo modprobe gs_usb
if [ $? -ne 0 ]; then
    echo "Error: Failed to load gs_usb module."
    exit 1
fi

# Check if only one CAN module needs to be handled
if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    if [ -n "$USB_ADDRESS" ]; then
        echo "Detected USB hardware address parameter: $USB_ADDRESS"

        # Find CAN interface corresponding to USB hardware address using ethtool
        INTERFACE_NAME=""
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
                INTERFACE_NAME="$iface"
                break
            fi
        done

        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: Cannot find CAN interface corresponding to USB address $USB_ADDRESS."
            exit 1
        else
            echo "Found interface corresponding to USB address $USB_ADDRESS: $INTERFACE_NAME"
        fi
    else
        # Get the unique CAN interface
        INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')

        # Check if interface name was obtained
        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: Cannot detect CAN interface."
            exit 1
        fi

        echo "Expected one CAN module, detected interface $INTERFACE_NAME"
    fi

    # Check if interface is already up
    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

    # Get current interface bitrate
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

    if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already up with bitrate $DEFAULT_BITRATE"

        # Check if interface name matches default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and brought back up."
        else
            echo "Interface name is already $DEFAULT_CAN_NAME"
        fi
    else
        # Configure interface if not up or bitrate differs
        if [ "$IS_LINK_UP" == "yes" ]; then
            echo "Interface $INTERFACE_NAME is up but bitrate $CURRENT_BITRATE differs from set $DEFAULT_BITRATE."
        else
            echo "Interface $INTERFACE_NAME is not up or bitrate not set."
        fi

        # Set interface bitrate and bring up
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
        sudo ip link set "$INTERFACE_NAME" up
        echo "Interface $INTERFACE_NAME reconfigured with bitrate $DEFAULT_BITRATE and brought up."

        # Rename interface to default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and brought back up."
        fi
    fi
else
    # Handle multiple CAN modules

    # Check if predefined USB ports count matches expected CAN modules count
    PREDEFINED_COUNT=${#USB_PORTS[@]}
    if [ "$EXPECTED_CAN_COUNT" -ne "$PREDEFINED_COUNT" ]; then
        echo "Error: Expected CAN modules ($EXPECTED_CAN_COUNT) does not match predefined USB ports ($PREDEFINED_COUNT)."
        exit 1
    fi

    # Iterate through all CAN interfaces
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        # Get bus-info using ethtool
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')

        if [ -z "$BUS_INFO" ];then
            echo "Error: Cannot get bus-info for interface $iface."
            continue
        fi

        echo "Interface $iface is on USB port $BUS_INFO"

        # Check if bus-info is in predefined USB ports list
        if [ -n "${USB_PORTS[$BUS_INFO]}" ];then
            IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[$BUS_INFO]}"

            # Check if interface is already up
            IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")

            # Get current interface bitrate
            CURRENT_BITRATE=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+')

            if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$TARGET_BITRATE" ]; then
                echo "Interface $iface is already up with bitrate $TARGET_BITRATE"

                # Check if interface name matches target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and brought back up."
                else
                    echo "Interface name is already $TARGET_NAME"
                fi
            else
                # Configure interface if not up or bitrate differs
                if [ "$IS_LINK_UP" == "yes" ]; then
                    echo "Interface $iface is up but bitrate $CURRENT_BITRATE differs from target $TARGET_BITRATE."
                else
                    echo "Interface $iface is not up or bitrate not set."
                fi

                # Set interface bitrate and bring up
                sudo ip link set "$iface" down
                sudo ip link set "$iface" type can bitrate $TARGET_BITRATE
                sudo ip link set "$iface" up
                echo "Interface $iface reconfigured with bitrate $TARGET_BITRATE and brought up."

                # Rename interface to target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and brought back up."
                fi
            fi
        else
            echo "Error: Unknown USB port $BUS_INFO for interface $iface."
            exit 1
        fi
    done
fi

echo "All CAN interfaces successfully renamed and activated."
