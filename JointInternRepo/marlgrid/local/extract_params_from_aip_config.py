from absl import app
from absl import flags

import yaml


flags.DEFINE_string('filepath', '', 'aip yaml filepath')

FLAGS = flags.FLAGS

def extract_param(param):
    def extract_param_value(param):
        if param['type'] == 'CATEGORICAL':
            values = param['categoricalValues']
            assert len(values) == 1
            return values[0]
        elif param['type'] in ['INTEGER', 'DOUBLE']:
            values = [param['minValue'], param['maxValue']]
            assert values[0] == values[1]
            return values[0]
        else:
            raise Exception('Unsupported param:' + str(param))
    return (param['parameterName'], extract_param_value(param))

def param_list_to_string(param_list):
    result = ''
    for pn, pv in param_list:
        assert '\'' not in str(pn)
        assert '\'' not in str(pv)
        result += '--' + str(pn) + '=\'' + str(pv) + '\' '
    return result

def main(argv):
    filepath = FLAGS.filepath
    with open(filepath) as f:
        data = yaml.load(f, Loader=yaml.Loader)
    params = data['trainingInput']['hyperparameters']['params']
    result = []
    for p in params:
        result.append(extract_param(p))
    print(param_list_to_string(result))

if __name__ == '__main__':
    app.run(main)
