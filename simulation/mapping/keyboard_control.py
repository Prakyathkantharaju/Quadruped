from pynput import keyboard




class KeyboardControl:

    def __init__(self):
        self.key_press = None
        self.key_release = None

    def on_press(self, key):
        try:
            self.key_press = key.char
        except AttributeError:
            self.key_press = None

    def on_release(self, key):
        try:
            self.key_release = key.char
        except AttributeError:
            self.key_release = None



    def main(self):
        # ...or, in a non-blocking fashion:
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()


    