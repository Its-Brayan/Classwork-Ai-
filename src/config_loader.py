import yaml
def load_config(filename="config.yaml"):
    config_path = f"/home/brayan/Aiprojects/ClassworkAi/config/{filename}"
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
        return config
def load_prompt(filename="prompt_config.yaml"):
    prompt_path = f"/home/brayan/Aiprojects/ClassworkAi/config/{filename}"
    with open(prompt_path,"r") as file:
        prompt_config = yaml.safe_load(file)
        return prompt_config