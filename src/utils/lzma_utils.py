import os
import lzma
import xml.etree.ElementTree as ET


def convert_lzma_file_to_xml(file_path: str, xml_root_dir: str):
    """
    Args:
    - file_path (str): path to an xml file compressed as lzma
    - xml_root_dir (str): directory where xml files should be saved
    Process and outcome:
    - Opens the .lzma file, uncompresses it and saves it in the xml_root_dir directory
    """
    if not os.path.exists(xml_root_dir):
        os.mkdir(xml_root_dir)
    with lzma.open(file_path, "rb") as compressed_file:
        xml_bytes = compressed_file.read()
    new_file_name = file_path[:-5]
    new_file_path = os.path.join(xml_root_dir, new_file_name)
    with open(new_file_path, "wb") as xml_file:
        xml_file.write(xml_bytes)