import fitz
import pandas as pd
import os



def get_path():
    file_path = []
    path1 = input("Enter the path of the AI files: ")
    print('Path registered successfully')
    path2 = input("Enter the path for WEB files: ")
    print('Path registered successful')
    file_path.append(path1)
    file_path.append(path2)
    return file_path



def get_final_df(path, flag):
    content = []
    df = pd.DataFrame(columns=["Text", "Label"])

    for file in os.listdir(path):
        if file.endswith('.pdf'):
            doc = fitz.open(path + '\\' + file)
            content_temp = ''
            for  page in range(len(doc)):
                content_temp = content_temp + doc[page].get_text()
            content.append(content_temp)

    df["Text"] = content
    df['Label'] = flag
    
    return df


def get_contents_of_pdfs(file_path):
    for path in file_path:
        if '\\AI' in path:
            df_ai = get_final_df(path, 1)
        elif '\\WEB' in path:
            df_web = get_final_df(path, 0)
    df = pd.concat([df_ai, df_web], axis=0)
    return df

def get_content(file_path):
    # df = pd.DataFrame(columns=["Text", 'Label']) for making code readable
    df = get_contents_of_pdfs(file_path)
    return df


def dataset_generate():
    file_path = get_path()
    dataset = get_content(file_path)
    dataset.to_csv('Dataset.csv')


if __name__ == "__main__":
    dataset_generate()
