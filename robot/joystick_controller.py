# Controller Class
from typing import Tuple

import pygame

class JoystickControl(object):
    def __init__(self, axis_range=2, axis_scale=3.0, dbg=False):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND, self.AXIS_RANGE, self.AXIS_SCALE = 0.1, axis_range, axis_scale
        self.dbg = dbg

    def input(self):
        pygame.event.get()
        r_joy = []

        # Latent Actions / 2D End-Effector Control
        if self.AXIS_RANGE == 2:
            for i in range(3, 3 + self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                r_joy.append(z * self.AXIS_SCALE)


# Secret, Tri-Axial End Effector Control
        else:
            for i in range(self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                r_joy.append(z * self.AXIS_SCALE)

        # Button Press
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)

        if self.dbg:
            print(f"r_joy : {r_joy}")
            print(f"a : {a}")
            print(f"b : {b}")
            print(f"x : {x}")
            print(f"y : {y}")

        return r_joy, a, b, x, y, stop

class Buttons(object):
    def __init__(self) -> None:
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self) -> Tuple[bool, bool, bool]:
        # Get "A", "B", "X", "Y" Button Presses
        pygame.event.get()
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y = self.gamepad.get_button(2), self.gamepad.get_button(3)

        return a, b, x, y

if __name__ == "__main__":
    js = JoystickControl(dbg=True)
    while True:
        r_joy, a, b, x, y, stop = js.input()
        if x:
            break
