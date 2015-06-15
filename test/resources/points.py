__author__ = "Polyarnyi Nickolay"


x, y, z = 29.5, 29.5, 0#10.0
width, height = 2592, 1936

points = {'up_left_bottom':    [0, 0, z],
          'up_left_top':       [0, y, z],
          'up_right_top':      [x, y, z],
          'up_right_bottom':   [x, 0, z],
          # 'down_left_bottom':  [0, 0, 0],
          # 'down_left_top':     [0, y, 0],
          # 'down_right_top':    [x, y, 0],
          # 'down_right_bottom': [x, 0, 0],
          }

observations = {
    'IMG_0644': {
        'up_left_bottom':    [ 356,  693],
        'up_left_top':       [1136,  192],
        'up_right_top':      [2027,  480],
        'up_right_bottom':   [1336, 1225],
        'down_left_bottom':  [ 486,  994],
        'down_right_top':    [1971,  784],
        'down_right_bottom': [1354, 1518],
    },
    'IMG_0645': {
        'up_left_bottom':    [ 536,  961],
        'up_left_top':       [ 885,  270],
        'up_right_top':      [1895,  336],
        'up_right_bottom':   [1974, 1085],
        'down_left_bottom':  [ 627, 1265],
        'down_right_bottom': [1890, 1387],
    },
}
