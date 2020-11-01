import pandas as pd

import numpy as np

import os

import cv2

from pycm import ConfusionMatrix

import lwmw


def pixel_accuracy(predict, true_mask):
    """Метрика попиксильной точности
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение метрики попиксильной точности
    """
    return np.sum(predict * true_mask) / (np.sum(true_mask))

def Jaccard(predict, true_mask):
    """Мера Жаккара
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Жаккара точности
    """
    return sum(predict * true_mask) / (sum(predict) + sum(true_mask) - sum(predict * true_mask))

def Sorensen(predict, true_mask):
    """Мера Соренсена
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Соренсена точности
    """
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask))

def Kulchinski(predict, true_mask):
    """Мера Кульчински
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Кульчински точности
    """
    return ((sum(predict * true_mask)) / 2) * (1 / sum(predict) + 1 / sum(true_mask))

def Simpson(predict, true_mask):
    """Мера Симпсона
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Симпсона точности
    """
    
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask) - abs(sum(predict) - sum(true_mask)))

def Braun(predict, true_mask):
    """Мера Брауна
    
    Arguments:
        predict (np.ndarray): массив с информацией о пикселях размеченного ИНС изображения (загруженного методом _raed_data_folder())
        predict (np.ndarray): массив с информацией о пикселях размеченного экспертом изображения (загруженного методом _raed_data_folder())
    Returns:
        metric (float): значение меры Брауна точности
    """
    
    return (2 * sum(predict * true_mask)) / (sum(predict) + sum(true_mask) + abs(sum(predict) - sum(true_mask)))

def get_metrics_table(path_to_preds, path_to_true, norm=255.0):
    """Получение таблицы (pandas.DataFrame) метрик на основе сегментированных данных ИНС и эксперта
    
    Arguments:
        path_to_preds (str): путь до папки с изображениями, размеченными ИНС (без знака / в конце строки, "/home/data")
        path_to_true (str): путь до папки с изображениями, размеченными экспертом (без знака / в конце строки, "/home/data"
        norm (float): значение нормировки изображений (Изображение / norm), default: 255.0
    Returns:
        metrics (pandas.DataFrame): таблица со значениями рассчитаных метрик
    """
    img_pred_data = []
    names_pred = sorted(list(os.listdir(path_to_preds + "/")))
    for img_pred_name in names_pred:
        img_pred_data.append(cv2.imread(f"{path_to_preds}/{img_pred_name}") / norm)
        
    img_true_data = []
    names_true = sorted(list(os.listdir(path_to_true + "/")))
    for img_true_name in names_true:
        img_true_data.append(cv2.imread(f"{path_to_true}/{img_true_name}") / norm)
    
    
    metrics = pd.DataFrame()
    for i in range(len(img_pred_data)):
        temp_overal = ConfusionMatrix(actual_vector=img_true_data[i].ravel(), predict_vector=img_pred_data[i].ravel())
        temp_metrics = {  "name": names_pred[i],
                          "PA": pixel_accuracy(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                          "Jaccard": Jaccard(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                          "Sorensen": Sorensen(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                          "Kulchinski": Kulchinski(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                          "Simpson": Simpson(img_pred_data[i].ravel(), img_true_data[i].ravel()),
                          "Braun_Blanke": Braun(img_pred_data[i].ravel(), img_true_data[i].ravel())
                       }
        
        temp_overal = temp_overal.overall_stat
        for j in temp_overal.keys():
            temp_metrics[j] = temp_overal[j]
    
        metrics = metrics.append(temp_metrics, ignore_index=True)
    
    #metrics = metrics.drop(columns=["95% CI", "P-Value", "Kappa 95% CI", "Overall J", "SOA4(Cicchetti)", "SOA1(Landis & Koch)", "SOA2(Fleiss)", "SOA3(Altman)", "SOA4(Cicchetti)", "SOA5(Cramer)", "SOA6(Matthews)", "Zero-one Loss", "Chi-Squared DF", "RR", "Lambda B", "Overall MCEN"])
    metrics = metrics[["name", "ACC Macro", "Bangdiwala B", "Bennett S", "Conditional Entropy", "Cross Entropy", "F1 Micro", "FNR Micro", "FPR Micro", "Gwet AC1","Hamming Loss", "Joint Entropy", "Kappa No Prevalence", "Mutual Information", "NIR", "Overall ACC", "Overall RACC", "Overall RACCU", "PPV Micro", "Reference Entropy", "Response Entropy", "Standard Error", "TNR Micro", "TPR Micro"]]
    
    #metrics = metrics.replace(to_replace='None', value=np.nan).dropna(axis="columns")
    
    metrics = metrics.set_index("name")
    return metrics
    
def predict(metrics, path_to_weights):
    """Оценка разметки ИНС
    
    Arguments:
        metrics (pandas.DataFrame): таблица метрик (из метода _get_metrics_table())
        path_to_weights (str): путь до файла с весами персептрона
        device (str): используемое устройство для вычисления default: "/gpu:1"
    Returns:
        metrics (pandas.Series): ряд с предсказаниями оценки эксперта
        grads (dict): данные для отрисовки графика значимости мер
    """
    
    settings = {
        "outs" : 5,
        "input_len" : len(metrics),
        "architecture" : [31,18],
        "inputs" : len(metrics.columns),
        "activation" : "sigmoid"
    }
    
    p = np.load(path_to_weights)

    predicts, grads = lwmw.predict(p, settings, metrics.values)
    
    for i in range(0, settings["outs"]):
        metrics["preds_" + str(i+1)] = predicts[:,i]
        
    metrics["pred"] = np.argmax(metrics[["preds_1", "preds_2", "preds_3", "preds_4", "preds_5"]].values, axis=1) + 1
    
    grads = np.sqrt(np.sum(grads[0]**2, axis=0) / len(grads[0])) / np.sqrt(np.sum(grads[0]**2, axis=0) / len(grads[0])).max()
    
    return metrics["pred"], grads

# Пример запуска для получения оценки:

#Получение метрик
# Sample_1
#metrics = get_metrics_table("C:/Devs/medhack/Dataset/sample_1", "C:/Devs/medhack/Dataset/Expert", norm=255.0)
#print(len(metrics.columns))
#metrics.to_csv("C:/Devs/medhack/test_1.csv", index=True)
#print("Метрики для 1 ИНС посчитаны")

# Sample_2
#metrics = get_metrics_table("C:/Devs/medhack/Dataset/sample_2", "C:/Devs/medhack/Dataset/Expert", norm=255.0)
#metrics.to_csv("C:/Devs/medhack/test_2.csv")
#print("Метрики для 2 ИНС посчитаны")

# Sample_3
#metrics = get_metrics_table("C:/Devs/medhack/Dataset/sample_3", "C:/Devs/medhack/Dataset/Expert", norm=255.0)
#metrics.to_csv("C:/Devs/medhack/test_3.csv")
#print("Метрики для 3 ИНС посчитаны")

# Загружаем локальные метрики для проверки ошибки обучения
#metrics = pd.read_csv("C:/Devs/medhack/test_3.csv")
#metrics = metrics.set_index("name")
open_part = pd.read_csv("C:/Devs/medhack/Dataset/OpenPart.csv")
#f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
#expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
#metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
#metrics = metrics[metrics.index.isin(expert_index)]
#metrics = metrics.sort_index()
#
#open_part.Case = open_part.Case.str.split("_").str[0] + "_" + open_part.Case.str.split("_").str[1].str.split(".").str[0]
open_part = open_part.set_index("Case")
open_part = open_part.sort_index()
#
#metrics["expert"] = open_part["Sample 3"]
#metrics = metrics.join(pd.get_dummies(metrics['expert']))
#metrics = metrics.drop(columns=["expert"]).rename(columns={1 : "mark_1", 2 : "mark_2", 3 : "mark_3", 4 : "mark_4", 5 : "mark_5"})
#


for i in range(1, 4):
    metrics = pd.read_csv(f"C:/Devs/medhack/data_{i}.csv")
    metrics = metrics.set_index("name")

    result, grads = predict(metrics[metrics.columns[:-5]], "C:/Devs/medhack/p_23.npy")

    expert_mark = open_part[f"Sample {i}"].values

    err = 0
    for i in range(len(result.values)):
        err += np.abs(expert_mark[i] - result.values[i])
        
    err = err / len(result)    
    print(f"Sample {i}, ошибка обучения (MAE): {err}")

print("\n\n")

# Загружаем локальные метрики (для теста) и предсказываем / отсекаем размеченные данные

metrics = pd.read_csv("C:/Devs/medhack/test_1.csv")
metrics = metrics.set_index("name")
open_part = pd.read_csv("C:/Devs/medhack/Dataset/OpenPart.csv")
f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
metrics = metrics[~metrics.index.isin(expert_index)]
metrics = metrics.sort_index()
main_result, grads = predict(metrics, "C:/Devs/medhack/p_23.npy")
main_result = main_result.to_frame().rename(columns={"pred":"Sample 1"})

for i in range(2, 4):
    metrics = pd.read_csv(f"C:/Devs/medhack/test_{i}.csv")
    metrics = metrics.set_index("name")
    open_part = pd.read_csv(f"C:/Devs/medhack/Dataset/OpenPart.csv")
    f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
    expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
    metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
    metrics = metrics[~metrics.index.isin(expert_index)]
    metrics = metrics.sort_index()

    result, grads = predict(metrics, "C:/Devs/medhack/p_23.npy")
    main_result = main_result.merge(result.to_frame().rename(columns={"pred":f"Sample {i}"}), left_index=True, right_index=True)
    
main_result.to_csv("result.csv")