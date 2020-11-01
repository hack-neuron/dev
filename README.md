# dev

Для получения разметки используется скрипт `neuramark.py`. Коментарии внутри скрипта дублируются.

Получение метрик долгая операция -- метрики заранее сформированы в `csv` файлы.

## Sample_1

```
metrics = get_metrics_table("<путь до папки>/Dataset/sample_1", "<путь до папки>/Expert", norm=255.0)
print(len(metrics.columns))
metrics.to_csv("<путь до папки>/test_1.csv", index=True)
print("Метрики для 1 ИНС посчитаны")
```

## Sample_2

```
metrics = get_metrics_table("<путь до папки>/Dataset/sample_2", "<путь до папки>/Expert", norm=255.0)
metrics.to_csv("C:/Devs/medhack/test_2.csv")
print("Метрики для 2 ИНС посчитаны")
```

## Sample_3

```
metrics = get_metrics_table("<путь до папки>/Dataset/sample_3", "<путь до папки>/Expert", norm=255.0)
metrics.to_csv("<путь до папки>/test_3.csv")
print("Метрики для 3 ИНС посчитаны")
```

## Загружаем локальные метрики для проверки ошибки обучения (повторить для 3х sample)

```
metrics = pd.read_csv("<путь до папки>/test_3.csv")
metrics = metrics.set_index("name")
open_part = pd.read_csv("<путь до папки>/OpenPart.csv")
f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
metrics = metrics[metrics.index.isin(expert_index)]
metrics = metrics.sort_index()

open_part.Case = open_part.Case.str.split("_").str[0] + "_" + open_part.Case.str.split("_").str[1].str.split(".").str[0]
open_part = open_part.set_index("Case")
open_part = open_part.sort_index()

metrics["expert"] = open_part["Sample 3"]
metrics = metrics.join(pd.get_dummies(metrics['expert']))
metrics = metrics.drop(columns=["expert"]).rename(columns={1 : "mark_1", 2 : "mark_2", 3 : "mark_3", 4 : "mark_4", 5 : "mark_5"})
```

### Проверка MAE на обучающей выборке

```
for i in range(1, 4):
    metrics = pd.read_csv(f"<путь до папки>/data_{i}.csv")
    metrics = metrics.set_index("name")

    result, grads = predict(metrics[metrics.columns[:-5]], "<путь до папки>/p_23.npy")

    expert_mark = open_part[f"Sample {i}"].values

    err = 0
    for i in range(len(result.values)):
        err += np.abs(expert_mark[i] - result.values[i])
        
    err = err / len(result)    
    print(f"Sample {i}, ошибка обучения (MAE): {err}")

print("\n\n")
```

## Загружаем локальные метрики (для теста) и предсказываем / отсекаем размеченные данные

```
metrics = pd.read_csv("<путь до папки>/test_1.csv")
metrics = metrics.set_index("name")
open_part = pd.read_csv("<путь до папки>/OpenPart.csv")
f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
metrics = metrics[~metrics.index.isin(expert_index)]
metrics = metrics.sort_index()
main_result, grads = predict(metrics, "<путь до папки>/p_23.npy")
main_result = main_result.to_frame().rename(columns={"pred":"Sample 1"})

for i in range(2, 4):
    metrics = pd.read_csv(f"<путь до папки>/test_{i}.csv")
    metrics = metrics.set_index("name")
    open_part = pd.read_csv(f"<путь до папки>/OpenPart.csv")
    f = [x.split("_")[0] + "_" + x.split("_")[1].split(".")[0] for x in list(open_part.Case.values)]
    expert_index = list(set(metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]) & set(f))
    metrics.index = metrics.index.str.split("_").str[0] + "_" + metrics.index.str.split("_").str[1]
    metrics = metrics[~metrics.index.isin(expert_index)]
    metrics = metrics.sort_index()

    result, grads = predict(metrics, "<путь до папки>/p_23.npy")
    main_result = main_result.merge(result.to_frame().rename(columns={"pred":f"Sample {i}"}), left_index=True, right_index=True)

main_result.to_csv("<путь до папки>/result.csv")
```
