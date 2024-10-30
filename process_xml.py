import os
import openai
import pandas as pd
from xml.etree import ElementTree as ET

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = ''

def read_xml_file(xml_file):
    try:
        with open(xml_file, 'r') as file:
            xml_data = file.read()
            return xml_data
    except FileNotFoundError:
        print(f"Error: File '{xml_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return None

def generate_documentation_from_xml(xml_data):
    prompt = (
        "You are a technical documentation expert tasked with analyzing business rules from an XML file. The XML data contains business rules related to ServiceNow. Your task is to create documentation with clear descriptions of the purpose of each business rule function present in the XML. Ensure each function is analyzed comprehensively, considering any conditions and scripts present.\n\n"
        "Create documentation in a tabular structure for each business rule function found within the XML. Avoid duplicating descriptions for the same function and ensure no data from the file is overlooked. Provide accurate and concise descriptions for all functions present.\n\n"
        "The structure for the table shall include: Function Name | Description\n\n"
        "| Function Name        |Sys_Id    | Description |"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
            top_p=1.0,
            n=1,
            stop=None
        )

        documentation = response.choices[0].message['content'].strip()
        print(f"Generated documentation content:\n{documentation}")  # Print the entire response
        return documentation
    except Exception as e:
        print(f"Error generating documentation: {e}")
        return None

def save_to_excel(documentation, filename):
    if documentation:
        rows = [row.strip() for row in documentation.split('\n') if row.strip()]
        data = []
        for row in rows:
            columns = [col.strip() for col in row.split('|') if col.strip()]
            if len(columns) == 2:
                data.append(columns)
        df = pd.DataFrame(data, columns=['Function Name', 'Description'])
        excel_filename = f"{filename}.xlsx"
        df.to_excel(excel_filename, index=False)
        print(f"Excel file saved: {excel_filename}")
    else:
        print("No documentation to save.")

if __name__ == '__main__':
    xml_folder_path = "C:/Users/JatinSahni/OneDrive - inmorphis.com/Desktop/OpenAI/Pinecone/data"  # Replace with the path to your XML folder

    for filename in os.listdir(xml_folder_path):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(xml_folder_path, filename)
            xml_data = read_xml_file(xml_file_path)
            if xml_data:
                documentation = generate_documentation_from_xml(xml_data)
                if documentation:
                    print(f"Generated Documentation for {filename}:")
                    print(documentation)
                    save_to_excel(documentation, filename)
                else:
                    print(f"Failed to generate documentation for {filename}.")
            else:
                print(f"Failed to read XML file: {xml_file_path}")
