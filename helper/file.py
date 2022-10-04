import json
import os

def load_json_file(json_file):
    """
    Load json file

    Return: json obj
    """
    
    file = open(json_file)
    return json.load(file)

def write_to_json_file(json_obj, json_file):
    """
    Write object json to json file

    Return json_file
    """

    if not os.path.exists(json_file):
        os.mknod(json_file)

    with open(json_file, "w") as outfile:
        json.dump(json_obj, outfile)

    return json_file
