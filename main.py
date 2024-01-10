import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


model_path = 'Models/'

# Загрузка датасета
cs_data = pd.read_csv('Data/csgo.csv')
X = cs_data.drop(['bomb_planted_True'], axis=1)
y = cs_data['bomb_planted_True']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка моделей
def models():
    model1 = pickle.load(open(model_path + 'model_ml1.pkl', 'rb'))
    model2 = pickle.load(open(model_path + 'model_ml2.pkl', 'rb'))
    model3 = XGBClassifier()
    model3.load_model(model_path + 'model_ml3.json')
    model4 = pickle.load(open(model_path + 'model_ml4.pkl', 'rb'))
    model5 = pickle.load(open(model_path + 'model_ml5.pkl', 'rb'))
    model6 = load_model(model_path + 'model_ml6.h5')
    return model1, model2, model3, model4, model5, model6

# Название страницы
st.title('Расчётно графичесикая работа ML')
st.header("Тема РГР:")
st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

# Навигация
st.sidebar.title('Навигация:')
page = st.sidebar.selectbox("Выберите страницу", ["Разработчик", "Датасет", "Визуализация", "Инференс модели"])

# Информация о разработчике
def page_developer():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Цику Георгий Русланович")
        st.write("Номер учебной группы: ФИТ-221")
        st.write("Увлечения: Программирование, занятия баскетболом")

    with col2:
        st.header("Фотография")
        st.image("me.jpg", width=200)

# Информаиця о нашем датасете
def page_dataset():
    st.title("Информация о наборе данных")

    st.markdown("""
    ### Описание Датасета CS:GO
    
    Файл датасета: csgo.csv

    Описание:
    Данный датасет содержит статистическую информацию о матчах в компьютерной игре Counter-Strike: Global Offensive. Содержит следующие столбцы:

    - index: Индекс записи.
    - time_left: Время до конца раунда.
    - ct_score: Счёт спецназовцев.
    - t_score: Счёт террористов.
    - ct_health: Здоровье спецназовцев.
    - t_health: Здоровье террорирстов.
    - ct_armor: Броня спецназовцев.
    - t_armor: Броня террористов.
    - ct_money: Деньги спецназовцев.
    - t_money: Деньги террорирстов.
    - ct_helmets: Шлемы у спецназовцев.
    - t_helmets: Шлемы у террорирстов.
    - ct_defuse_kits: Набора сапёра у спецназовцев.
    - ct_players_alive: Живых спецназовцев.
    - t_players_alive: Живых террористов.
    - bomb_planted_True: Индикатор установленной бомбы.
                
    ### Особенности предобработки данных:
    - Удаление лишних столбцов (index).
    - Обработка пропущенных значений.
    - Нормализация числовых данных для улучшения производительности моделей.
    - Кодирование категориальных переменных.
    """)

# Страница с визуализацией
def page_data_visualization():
    st.title("Визуализации данных CS:GO")

    # Визуализация 1: Сравнение счёта команд
    fig, ax = plt.subplots()
    sns.boxplot(data=cs_data[['ct_score', 't_score']], ax=ax)
    ax.set_title('Счёт команд')
    st.pyplot(fig)

    # Визуализация 2: Количество денег у обеих команд
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=cs_data, x='index', y='ct_money', label='Деньги спецназа', ax=ax)
    sns.lineplot(data=cs_data, x='index', y='t_money', label='Деньги террористов', ax=ax)
    ax.set_title('Количество денег у команд в разных раундах')
    ax.set_xlabel('Номер раунда')
    ax.set_ylabel('Деньги')
    ax.legend()
    st.pyplot(fig)

    # Визуализация 3: Распределение здоровья команд
    fig, ax = plt.subplots()

    sns.histplot(cs_data['ct_health'], kde=True, label='Здоровье спецназа', ax=ax)
    sns.histplot(cs_data['t_health'], kde=True, label='Здьоровье террорирстов', ax=ax)
    ax.set_title('Здоровье обеих команд: ')
    ax.legend()
    st.pyplot(fig)


    # Визуализация 4: Общий уровень брони команд
    fig, ax = plt.subplots()
    sns.kdeplot(cs_data['ct_armor'], shade=True, label='Броня спецназа', ax=ax)
    sns.kdeplot(cs_data['t_armor'], shade=True, label='Броня террорирстов', ax=ax)
    ax.set_title('Броня у обеих команд: ')
    ax.legend()
    st.pyplot(fig)


# Страница с инференсом моделей
def page_predictions():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        feature_names = ['index','time_left','ct_score','t_score','ct_health','t_health','ct_armor','t_armor','ct_money','t_money','ct_helmets','t_helmets','ct_defuse_kits','ct_players_alive','t_players_alive']
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100000.0, value=10.0)

        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model_ml1, model_ml2, model_ml3, model_ml4, model_ml5, model_ml6 = models()

            input_df = pd.DataFrame([input_data])
            
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml2 = model_ml2.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int)

            # Вывод результатов
            st.success(f"Результат предсказания LogisticRegression: {prediction_ml1[0]}")
            st.success(f"Результат предсказания KMeans: {prediction_ml2[0]}")
            st.success(f"Результат предсказания XGBClassifier: {prediction_ml3[0]}")
            st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
            st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_ml6[0]}")
    else:
        try:
            model_ml1 = pickle.load(open(model_path + 'model_ml1.pkl', 'rb'))
            model_ml2 = pickle.load(open(model_path + 'model_ml2.pkl', 'rb'))
            model_ml3 = XGBClassifier()
            model_ml3.load_model(model_path + 'model_ml3.json')
            model_ml4 = pickle.load(open(model_path + 'model_ml4.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_path + 'model_ml5.pkl', 'rb'))
            model_ml6 = load_model(model_path + 'model_ml6.h5')

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml2 = model_ml2.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round()

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = accuracy_score(y_test, predictions_ml2)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность KMeans: {accuracy_ml2}")
            st.success(f"Точность StackingClassifier: {accuracy_ml3}")
            st.success(f"Точность XGBClassifier: {accuracy_ml4}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml5}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")



if page == "Разработчик":
    page_developer()
elif page == "Датасет":
    page_dataset()
elif page == "Визуализация":
    page_data_visualization()
elif page == "Инференс модели":
    page_predictions()
