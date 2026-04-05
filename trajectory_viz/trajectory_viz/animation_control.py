"""
Keyboard controller for forklift animation.

  s — snap to start pose (no animation)
  e — snap to end pose   (no animation)
  r — run once (re-press to replay)
  l — continuous loop
  q — quit
"""
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


def _read_key():
    """Read a single character without waiting for Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = Node('animation_control')
    pub_loop       = node.create_publisher(Empty, '/animation/loop',       10)
    pub_run_once   = node.create_publisher(Empty, '/animation/run_once',   10)
    pub_goto_start = node.create_publisher(Empty, '/animation/goto_start', 10)
    pub_goto_end   = node.create_publisher(Empty, '/animation/goto_end',   10)

    print('Animation control ready.')
    print('  s — start pose')
    print('  e — end pose')
    print('  r — run once')
    print('  l — continuous loop')
    print('  q — quit')

    try:
        while rclpy.ok():
            key = _read_key()
            if key == 's':
                pub_goto_start.publish(Empty())
                print('>> start pose')
            elif key == 'e':
                pub_goto_end.publish(Empty())
                print('>> end pose')
            elif key == 'r':
                pub_run_once.publish(Empty())
                print('>> run once')
            elif key == 'l':
                pub_loop.publish(Empty())
                print('>> loop')
            elif key in ('q', '\x03'):   # q or Ctrl-C
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
