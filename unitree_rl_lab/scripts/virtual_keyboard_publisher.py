#!/usr/bin/env python3
import argparse
import os
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_ as WirelessControllerDefault,
)


def make_key_value(state: dict) -> int:
    key_map = {
        "R1": 0,
        "L1": 1,
        "start": 2,
        "select": 3,
        "R2": 4,
        "L2": 5,
        "F1": 6,
        "F2": 7,
        "A": 8,
        "B": 9,
        "X": 10,
        "Y": 11,
        "up": 12,
        "right": 13,
        "down": 14,
        "left": 15,
    }
    keys = 0
    for name, bit in key_map.items():
        v = 0
        if name in ("F1", "F2"):
            v = 0
        else:
            v = 1 if state.get(name, 0) else 0
        keys |= (v << bit)
    return keys


def main():
    parser = argparse.ArgumentParser(description="Virtual keyboard publisher for Unitree C++ bridge")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--hz", type=float, default=100.0)
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gui_path = os.path.join(root_dir, "unitree_mujoco", "simulate_python")
    if gui_path not in sys.path:
        sys.path.insert(0, gui_path)

    import pygame
    from virtual_controller_gui import VirtualControllerGUI

    ChannelFactoryInitialize(args.domain_id, args.interface)
    pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    pub.Init()

    gui = VirtualControllerGUI(width=800, height=420)
    gui.init_display()

    dt = 1.0 / max(args.hz, 1.0)
    print("Virtual keyboard publisher started.")
    print("Window: Virtual Controller")
    print("WASD/Arrow=Move, Q/E/Z/X=Turn, buttons: Stand/Walk/Stop")

    while gui.running:
        t0 = time.perf_counter()
        gui.process_events()
        gui.render()

        state = gui.get_state()
        msg = WirelessControllerDefault()
        msg.keys = make_key_value(state)
        msg.lx = float(state.get("lx", 0.0))
        msg.ly = float(state.get("ly", 0.0))
        msg.rx = float(state.get("rx", 0.0))
        msg.ry = float(state.get("ry", 0.0))
        pub.Write(msg)

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    pygame.quit()


if __name__ == "__main__":
    main()
