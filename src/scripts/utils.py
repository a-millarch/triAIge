import yaml
import mlflow


def load_configs(ymls):
    cfg={}

    # ymls keys will be variable names referencing configfiles for data,
    for var_name, yml in ymls.items():
            handle =  f"../configs/{yml}.yaml"
            with open(handle, 'r') as file:
                    cfg[var_name] =  yaml.safe_load(file)
    return cfg

def log_configs(ymls):
    # ymls keys will be variable names referencing configfiles for data,
    for yml in ymls.values():
            handle =  f"../configs/{yml}.yaml"
            mlflow.log_artifact(handle)
