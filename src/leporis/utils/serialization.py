__author__ = "Polyarnyi Nickolay"


def deserialize_object(clazz, state):
    if clazz.serial_code != state['serial_code']:
        raise SerialCodeMismatch(clazz.serial_code, state['serial_code'])
    obj = clazz.__new__(clazz)
    obj.__setstate__(state)
    return obj


def deserialize_list(clazz, states):
    objs = []
    for state in states:
        objs.append(deserialize_object(clazz, state))
    return objs


def deserialize_dict(clazz, states):
    objs = {}
    for key, value in states.items():
        objs[key] = deserialize_object(clazz, value)
    return objs


def serialize_object(obj):
    if isinstance(obj, list):
        states = []
        for item in obj:
            states.append(serialize_object(item))
        return states

    if isinstance(obj, dict):
        states = {}
        for key, value in obj.items():
            states[key] = serialize_object(value)
        return states

    state = obj.__getstate__()
    assert 'serial_code' not in state
    state['serial_code'] = obj.serial_code
    return state


class SerialCodeMismatch(Exception):

    def __init__(self, expected, found):
        super(SerialCodeMismatch, self).__init__('Serial code mismatch: expected={}, found={}.'.format(expected, found))
