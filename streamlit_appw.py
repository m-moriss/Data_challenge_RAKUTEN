import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image, ImageOps
import pandas as pd
import pathlib



from donnees import (
    load_dataset,
    process_text,
    #load_vgg16_cnn_model,
    get_image,
    load_logisitc_regression_model
)

import plotly.express as px
import numpy as np

# pip install streamlit-option-menu
# pip install beautifulsoup4 tensorflow

# Screens:
# * Introduction du projet
#   * Challenge Data ENS: Rakuten fourni des données propriétaires
#   * But du projet: classification multimodale
# * Le dataset:
#   * Explication des différents champs
#   * [Dynamique] Choix dune catégorie: on affiche aléatoirement des annonces avec pour chacune: titre, description, image associée, nom de la catégorie quon a utilisé
# * Exploration des données:
#   * Le texte:
#     * Affichage des différents barchart statiques
#     * [Dynamique] Choix dune catégorie: on affiche les nuages de mots, les histogrammes de fréquence
#   * Les images:
#     * [Dynamique] Choix dune catégorie: on affiche les images avec les boundings box des objects


title_color="#38a"

# Setter le titre de la page + favicon
with Image.open('ds.webp') as favicon:
    st.set_page_config(
        page_title="Classification d'annonces de produit",
        layout="wide",
        page_icon=favicon
    )


st.markdown("""
<style> 
  .page_title_1 { 
    font-size:35px ; 
    font-family: 'Cooper Black'; 
    color: #38a;
  }

  .page_title_2 { 
    font-size:25px ; 
    font-family: 'Cooper Black'; 
    color: #38a;
  }

  .page_title_3 { 
    font-size:20px ; 
    font-family: 'Cooper Black'; 
    color: #38a;
  }

  .page_text { 
    font-size:35px ; 
    font-family: 'Cooper Black'; 
    color: #38a;
  }

  .titre_annonce { 
    font-size:12px ; 
    font-family: 'Cooper Black'; 
    color: #444;
  }

  .description_annonce { 
    font-size:10px ; 
    font-family: 'Cooper Black'; 
    color: #888;
  }

  .titre_annonce_big { 
    font-size:18px ; 
    font-weight: bold;
    font-family: 'Cooper Black'; 
    color: #444;
  }


  .stTabs > div > div {
    scrollbar-width: 10px !important;
  }

</style> 
""", unsafe_allow_html=True)



X_train, y_train, categories, X, categories_numbered, categories_alphasort = load_dataset()
logistic_regression_model = load_logisitc_regression_model()
#vgg16_cnn_model = load_vgg16_cnn_model(len(categories))

##################################
# La barre de menu sur la gauche #
##################################

with st.sidebar:
    choose = option_menu(
        "Classification d'annonces de produits",
        [
            "Introduction au projet",
            "Le dataset",
            "Exploration des données",
            "Préprocessing des données",
            "Algorithmes sur le texte",
            "Algorithmes sur les images",
            "Algorithme multimodal",
            "Démonstration",
            "Bilan",
        ],
        icons=['play-btn', 'book', 'bar-chart', 'segmented-nav', 'body-text','images', 'toggles','bi bi-play-circle', 'flag'],
        menu_icon="app-indicator", default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "#48d", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#38a"},
        }
    )



st.sidebar.write("""
<div class="menu">
<div class="container-xxl" style="background-color: rgb(230, 230, 230); border-radius:0.5rem; margin: 0.5rem; padding:1rem;">
<span style="font-size:25px">Projet DataScience</span>

<span style="font-style:italic">Soutenance Projet Octobre 2022</span>

Participants:

<ul>
<li><a href="https://www.linkedin.com/in/martinemorisseau/">Martine MORISSEAU</a></li>
<li><a href="https://www.linkedin.com/in/remi-humbert-4088b316/">Rémi HUMBERT</a></li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)
    

##########################
# Introduction au projet #
##########################

if choose == "Introduction au projet":
    st.markdown('<p class="page_title_1">Introduction au projet</p>', unsafe_allow_html=True)
    st.write("""
Notre projet s’inscrit dans le cadre de la **formation continue DataScientest, parcours Data Scientist.** 

Ce projet, organisé par le Rakuten Institute of Technology, fait également partie de la liste des sujets soumis au Challenge Data de 2020 et concerne la **classification des données des produits multimodaux.**

Notre objectif est de **construire un système de classification automatique des annonces du site web Rakuten.**

""")

    with Image.open(r'annonce.webp') as example_annonce:
        col1, col2, col3 = st.columns([1, 5, 1])
        col2.image(example_annonce, use_column_width=True, caption="Exemple d'annonce de produit Rakuten")

    st.write("""
A partir des données de l'annonce telles l'image, le titre et la description, notre système devra prédire la catégorie générale du produit.
""")

    texte_annonce = Image.open(r'texte_annonce.webp')
    image_annonce = Image.open(r'image_annonce.webp')
    cols = st.columns([1, 1, 1, 5, 1])
    cols[1].image(image_annonce, use_column_width=True, caption="Image extraite de l'annonce")
    cols[3].image(texte_annonce, use_column_width=True, caption="Texte extrait de l'annonce")
    st.write("""
La classe que devra prédire notre système pour cette annonce de produit est 'accessoire jeux'.
""")

    

        
##############
# Le Dataset #
##############

if choose == "Le dataset":
    st.markdown('<p class="page_title_1">Le dataset</p>', unsafe_allow_html=True)
    st.write("""
Pour construire ce système de classification automatique nous avons à notre disposition un jeu de données extrait du site web Rakuten, composé des données textuelles et des images de 84916 annonces de produits. Ces annonces sont déjà classifiées, c'est-à-dire que pour chacune de ces annonces nous disposons d’un identifiant de catégorie unique. Grâce à ce jeu de données nous allons pouvoir utiliser des méthodes d’apprentissage automatique enfin d'entraîner des algorithmes à classifier automatiquement des annonces de produit.
""")
    st.markdown('<p class="page_title_2">Des données propriétaires</p>', unsafe_allow_html=True)

    st.write("""Les données sont disponibles sous licence non ouverte, et sont strictement confidentielles et la propriété du Rakuten Institute of Technology""")

    st.markdown('<p class="page_title_2">Les données brutes</p>', unsafe_allow_html=True)

    st.write("""
RAKUTEN a mis à disposition les jeux de données organisés de la façon suivante:
* Un fichier `X_train.csv` contenant les **données textes de 8 4916 annonces de produits**
""")

    xtrain_placeholder = st.empty()
    st.write("""
* Un fichier `y_train.csv` contenant les **catégories associées à chacune des 84 916 annonces.**
""")
    ytrain_placeholder = st.empty()
    st.write("""
* Un fichier `X_test.csv` contenant les données textes de produits non classifiés. **Ce jeu de données n'est pas labelisé. Il n'a pas été utilisé pour la phase de modélisation**
* Un dossier zippé images.zip composé de deux dossiers `image_train` et `image_test`  contenant chacun la liste des images pour les phases d'entraînement et de test. **La taille de ces dossiers est d'environ 2Go**
""")

    images_placeholder = st.empty()

    def draw_and_display_dataset_sample():
        sample = X_train.sample(5)
        xtrain_placeholder.write(sample)
        ytrain_placeholder.write(y_train.loc[sample.index])
        with images_placeholder.container():
            cols = st.columns([1, 1, 1, 1, 1])
            index=0
            for i, row in sample.iterrows():
                
                 filename = f"img_train/image_{row.imageid}_product_{row.productid}.webp"
                 image = Image.open(filename)
                 cols[index].image(image, use_column_width=True)
                 index += 1

    draw_and_display_dataset_sample()

    st.button("Redraw samples", on_click=draw_and_display_dataset_sample)


    
  
###########################
# Exploration des données #
###########################

if choose == "Exploration des données":
    st.markdown('<p class="page_title_1">Exploration des données</p>', unsafe_allow_html=True)

    st.write(""" L'objectif de cette phase est de :
- S'approprier le jeu de données (type, données manquantes, doublons)
- Identifier les traitements à réaliser (ajout de données, suppression, concaténation,etc.)
- Préparer les features pour la phase de modélisation
""")



    st.markdown('<br><p class="page_title_3">1. Exploration des codes et de leur mots-clés</p></br>', unsafe_allow_html=True)

    st.write("""
Pour simplifier l’analyse nous avons décidé d'étiqueter manuellement nos catégories en observant un échantillon jeu de donné manuellement.
Résultat : **nous ajouterons une variable nommé "catégorie" qui pourra faire office de variable cible à prédire.**
""")

    tabs = st.tabs(list(categories.prdlabelcode.array))

    nb_sample = 2
    for i in range(0, len(categories)):
        with tabs[i]:
            st.markdown(f"<p><strong>Exemples d'annonces de la catégorie {categories.iloc[i].prdlabelcode}</strong></p>", unsafe_allow_html=True)
            category = categories.iloc[i].prdtypecode
            sample = X[y_train.prdtypecode == category].sample(nb_sample)
            cols = st.columns([1]*nb_sample)
            index=0
            for indice, row in sample.iterrows():
            
                filename = f"img_train/image_{row.imageid}_product_{row.imageid}.webp"
                image = Image.open(filename)
                cols[index].image(image, use_column_width=True)
                cols[index].write(f'<p class="titre_annonce">{row.designation[:50]}</p>', unsafe_allow_html=True)
                if str(row.description) != "nan":
                    cols[index].write(row.description[:300])
                index += 1
            category_image = f'categories-{"{:02d}".format(i)}n.webp'
            st.markdown("<p><strong>Nuage de mot et fréquence des mots de la catégorie</strong></p>", unsafe_allow_html=True)    
            st.write("""Pour avoir une idée plus précise du contenu du texte des annonces par catégorie, **nous avons réalisé des nuages de mots et des histogrammes avec les mots les plus fréquents par catégorie.**
            """)
            st.image(category_image)

    st.markdown('<p class="page_title_3">2. Analyse des annonces par catégories et par langues</p>', unsafe_allow_html=True)

    df_langues = pd.read_csv('Repartition_langues.csv',sep=";")


    class_distr = st.container()

    class_distr.markdown('''
    Après avoir donné des labels à nos catégories, nous pouvons analyser la répartition des annonces par catégories.
    **Certaines catégories sont plus représentées** que d'autres.
    A tire d'exemple, nous avons environ 9000 annonces pour la catégorie "jardin, ameublement jardin" tandis que le volume d'annonces pour la catégorie "Billard, flechettes,autres"
    n'est que de 693.
    ''',  unsafe_allow_html=True)

    hist = px.histogram(X, x='categorie',
                               template='none',
                               color='categorie',
                               height=400,
                               width=800,
                               color_discrete_sequence=px.colors.sequential.Viridis,
                            ).update_xaxes(categoryorder="total ascending")
    class_distr.plotly_chart(hist)

    class_distr.markdown('''
    Plusieurs langues sont également présentes dans les descriptions des produits. Cela influencera notre méthode dans la phase de préprocessing :
    - Traduire l'ensemble des textes dans une unique langue ?
    - Traiter les descriptions par langues ?
    ''',  unsafe_allow_html=True)

    st.table(df_langues)


    st.markdown('<p class="page_title_3">3. Analyse des annonces par nombres de mots, présence de description</p>', unsafe_allow_html=True)
    st.write("""
Bien que toutes les annonces de produit aient un titre, 36% d'entre elles n'ont pas de description.
""")

    with Image.open(r'piechart_description.webp') as img:
        col1, col2, col3 = st.columns([5, 3, 5])
        col2.image(img, use_column_width=True)

    st.write("""
Et l'on observe de grosses disparités au sein des catégories, avec la catégorie Livres, mangas, romans contenant le moins de descriptions associées aux annonces.
""")
        
    with Image.open(r'barchart_description.webp') as img:
        col1, col2, col3 = st.columns([2, 6, 2])
        col2.image(img, use_column_width=True)

    st.write("""
Si l’on regarde le nombre de lettre qui composent les titres des annonces, il semble que toutes les annonces aient toutes des titres plutôt longs (médiane de 50 lettres par titre pour les catégories dont les titres ont le moins de lettres)
""")

        
    with Image.open(r'title_char.webp') as img:
        col1, col2, col3 = st.columns([2, 6, 2])
        col2.image(img, use_column_width=True)

    






            
#############################
# Préprocessing des données #
#############################


if choose == "Préprocessing des données":
    st.markdown('<p class="page_title_1">Préprocessing des données</p>', unsafe_allow_html=True)

    st.markdown('<p class="page_title_2">Préprocessing du texte</p>', unsafe_allow_html=True)

    st.write("""

Pour utiliser le texte dans les algorithmes de classification nous avons réalisé une étape de preprocessing afin que nos modèles puissent prendre en entrée les **mots les plus pertinents qui apportent de l’information.** 
Nous avons réalisé les actions suivantes:
<ul>
<li>Fusion des titres (designation) et des descriptions (description) en un seul champ texte</li>
<li>**Suppression des annonces en doublons**</li>
<li>Suppression des **balises html**</li>
<li>Suppression des **stop words** Francais, Anglais, des nombres de 1 à 100 et de la lettre X</li>
<li>Lemmatization</li>
<li>Tokenisation</li>
<li>Passage des descriptions en minuscule</li>
</ul>
""", unsafe_allow_html=True)
    

    st.markdown('<p class="page_title_3">Exemple</p>', unsafe_allow_html=True)

    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    placeholder4 = st.empty()
    
    def draw_sample():
        sample = X.sample(1)
        orig_texte = sample.texte.tolist()[0]
        #placeholder.write()
        texte, vector = process_text(orig_texte)
        texte_for_display = ', '.join(texte)
        placeholder1.write(f"""**Le texte original:**<br/>""", unsafe_allow_html=True)
        placeholder2.write(orig_texte)

        placeholder3.write(f"""<br/>**Le texte de l'annonce une fois préprocessé:**<br/>""", unsafe_allow_html=True)
        placeholder4.write(texte_for_display)

    draw_sample()

    st.button("Choisir un autre exemple", on_click=lambda: None)

    st.markdown('<p class="page_title_2">Préprocessing des images</p>', unsafe_allow_html=True)

    st.write("""
Les images ont été **réduites en résolution pour accelerer l'apprentissage des algorithmes.**
    """)
    
############################
# Algorithmes sur le texte #
############################

if choose == "Algorithmes sur le texte":
    st.markdown('<p class="page_title_1">Algorithmes sur le texte</p>', unsafe_allow_html=True)
    st.write("""
Explication dataframe après preprocessing
""")


    #df_Xtexte_origin = pd.read_csv('C:/Users/Morisseau1/AppData/Roaming/Python/Python310/site-packages/streamlit/streamlit/rakuten/df_Xtexte_origin.csv',sep=";",encoding='unicode_escape')
    df_Xtexte_origin = pd.read_csv('df_Xtexte_origin.csv',sep=";",encoding='unicode_escape')

    st.table(df_Xtexte_origin)


    #df_Xtexte_modifie = pd.read_csv('C:/Users/Morisseau1/AppData/Roaming/Python/Python310/site-packages/streamlit/streamlit/rakuten/df_Xtexte_modifie.csv',sep=";",encoding='unicode_escape')
    df_Xtexte_modifie = pd.read_csv('df_Xtexte_modifie.csv',sep=";",encoding='unicode_escape')
    st.table(df_Xtexte_modifie)

    st.markdown("<p class=\"page_title_2\">Modélisation</p>", unsafe_allow_html=True)

    st.write("""Nous avons testés plusieurs modèles appliqués aux données textes parmi des modèles de **machine learning et modèles de deep learning :**
""")

    #with Image.open('C:/Users/Morisseau1/AppData/Roaming/Python/Python310/site-packages/streamlit/streamlit/images/Listes_modeles_ML.png') as img:
    with Image.open('Listes_modeles_ML.PNG') as img:        
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.image(img, use_column_width=True)

    st.markdown("<p class=\"page_title_2\">Le meilleur modèle</p>", unsafe_allow_html=True)
    st.write("""
    Voici la matrice de confusion :
""")

    #with Image.open('C:/Users/Morisseau1/AppData/Roaming/Python/Python310/site-packages/streamlit/streamlit/images//Matrice_RNN1.png') as img:
    with Image.open('Matrice_RNN1.PNG') as img:
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.image(img, use_column_width=True)
        
        
##############################
# Algorithmes sur les images #
##############################

if choose == "Algorithmes sur les images":
    st.markdown('<p class="page_title_1">Algorithmes sur les images</p>', unsafe_allow_html=True)

    st.write("""Nous avons testés plusieurs modèles:
* Un reseau de neurone convolutif simple
* Plusieurs CNN basés sur VGG16
* Plusieurs CNN basés sur Xception
""")

    st.markdown("<p class=\"page_title_2\">Le meilleur modèle</p>", unsafe_allow_html=True)
    st.write("""
Le modele qui a donné les meilleurs résultats est basé sur VGG16.
Voici son architecture:
""")


    with Image.open('best_image.png') as img:
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.image(img, use_column_width=True)
    
    st.write("""
    Nous obtenons une accuracy de 0.5375 sur notre jeu de test, ainsi que la matrice de confusion suivante.""")


    with Image.open('image_accuracy.png') as img:
        col1, col2, col3 = st.columns([2, 6, 2])
        col2.image(img, use_column_width=True)

    
    with Image.open('image_vgg16.PNG') as img:
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.image(img, use_column_width=True)

    st.write("""
Et les f1-scores suivants:
""")


    with Image.open('vgg16_f1score.PNG') as img:
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.image(img, use_column_width=True)
        
        
#########################
# Algorithme multimodal #
#########################

if choose == "Algorithme multimodal":
    st.markdown('<p class="page_title_1">Algorithme multimodal</p>', unsafe_allow_html=True)

    st.markdown("<p class=\"page_title_2\">Méthode ensembliste avec moyenne pondérée</p>", unsafe_allow_html=True)

    st.write("""
Nous avons réuni les modèles grâce à une méthode ensembliste utilisant une moyenne pondérée, dont nous avons fixé les poids à la main, sans entraînement.
""")

    with Image.open(r'union_modele.png') as img:
        col1, col2, col3 = st.columns([2, 6, 2])
        col2.image(img, use_column_width=True)

    st.write("""
Nous avons fixés à la main les poids W_texte a 0.8 et W_image a 0.5 (les accuracy respectives de chaque modèle).  
""")

        
    st.write("""
Sur le jeu de données de test, le classifieur multimodal obtient une accuracy de 0.827, ainsi que la matrice de confusion suivante.
""")

    with Image.open(r'multimodal.webp') as img:
        col1, col2, col3 = st.columns([2, 6, 2])
        col2.image(img, use_column_width=True)

    st.write("""
Les f1-score sur chaque classe sont les suivants:
""")

    with Image.open(r'union_f1score.png') as img:
        col1, col2, col3 = st.columns([2, 3, 2])
        col2.image(img, use_column_width=True)

    st.markdown("<p class=\"page_title_2\">Analyse des classes mal prédites</p>", unsafe_allow_html=True)

        
    st.write("""
    Les classes les moins bien prédites (plus bas f1-score) sont
    * 1281-Jeux de société, confondus avec:
      * 1280-Jeux type playmobil en entrée
      * 1180-Figurines 2 en sortie
""")

    
    st.markdown(f"<p><strong>Exemples d'annonces de la catégorie 1281-Jeux de société</strong></p>", unsafe_allow_html=True)
    nb_sample=6
    sample = X[y_train.prdtypecode == 1281].sample(nb_sample)
    cols = st.columns([1]*nb_sample+[4])
    index=0
    for indice, row in sample.iterrows():
        filename = f"img_train/image_{row.imageid}_product_{row.productid}.jpg"
        #filename = f"C:/Users/Morisseau1/DSPP/Donnees/images (1)/images/image_train/image_{row.imageid}_product_{row.productid}.jpg"
        image = Image.open(filename)
        cols[index].image(image, use_column_width=True)
        cols[index].write(f'<p class="titre_annonce">{row.designation[:50]}</p>', unsafe_allow_html=True)
        #if str(row.description) != "nan":
        #    cols[index].write(row.description[:100])
        index += 1

    st.markdown(f"<p><strong>Exemples d'annonces de la catégorie 1280-Jeux type playmobil</strong></p>", unsafe_allow_html=True)
    nb_sample=6
    sample = X[y_train.prdtypecode == 1280].sample(nb_sample)
    cols = st.columns([1]*nb_sample+[4])
    index=0
    for indice, row in sample.iterrows():
        filename = f"img_train/image_{row.imageid}_product_{row.productid}.jpg"
        #filename = f"C:/Users/Morisseau1/DSPP/Donnees/images (1)/images/image_train/image_{row.imageid}_product_{row.productid}.jpg"
        image = Image.open(filename)
        cols[index].image(image, use_column_width=True)
        cols[index].write(f'<p class="titre_annonce">{row.designation[:50]}</p>', unsafe_allow_html=True)
        #if str(row.description) != "nan":
        #    cols[index].write(row.description[:100])
        index += 1

    st.markdown(f"<p><strong>Exemples d'annonces de la catégorie 1180-Figurines 2</strong></p>", unsafe_allow_html=True)
    nb_sample=6
    sample = X[y_train.prdtypecode == 1180].sample(nb_sample)
    cols = st.columns([1]*nb_sample+[4])
    index=0
    for indice, row in sample.iterrows():
        filename = f"img_train/image_{row.imageid}_product_{row.productid}.jpg"
        #filename = f"C:/Users/Morisseau1/DSPP/Donnees/images (1)/images/image_train/image_{row.imageid}_product_{row.productid}.jpg"
        image = Image.open(filename)
        cols[index].image(image, use_column_width=True)
        cols[index].write(f'<p class="titre_annonce">{row.designation[:50]}</p>', unsafe_allow_html=True)
        #if str(row.description) != "nan":
        #    cols[index].write(row.description[:100])
        index += 1
        
    st.write("""Il est difficile de faire la différence entre ces classes""")

#################
# Demonstration #
#################

if choose == "Démonstration":
    st.markdown("<p class=\"page_title_2\">Démonstration du classifieur</p>", unsafe_allow_html=True)

    cols = st.columns([3, 1, 7])

    
    categorie = cols[0].selectbox(
        '',
        list(categories.index),
        format_func=lambda option: (
            categories.loc[option].prdlabelcode if option else "<Selectionnez une catégorie>"
        )
    )
    #categorie=6

    if categorie:
        prdtypecode = categories.loc[categorie].prdtypecode
        annonce = cols[0].selectbox(
            '',
            [None] + list(X[X.prdtypecode == prdtypecode].index),
            format_func=lambda option: (
                X.loc[option].designation[:100] if option else "<Selectionnez une annonce>"
            )
        )
        #annonce=41
        
        if annonce:
            row = X.loc[annonce]
            cols[2].write(f'<br><p class="titre_annonce_big">{row.designation}</p>', unsafe_allow_html=True)
            cols[2].image(f"image_train/{row.image}")
            if str(row.description) != "nan":
                cols[2].write(row.description)


            st.markdown("<p class=\"page_title_3\">Classification du texte (Régression Logistique)</p>", unsafe_allow_html=True)

            texte, vector = process_text(row.texte)
            texte_for_display = ', '.join(texte)
            st.write(f"Le texte de l'annonce une fois préprocessé:<br><span style=\"font-style: italic;\">{texte_for_display}</span>", unsafe_allow_html=True)


            def display_top_proba(prediction):
                probas = pd.concat([
                    categories_alphasort.prdlabelcode,
                    pd.Series(np.round(prediction[0]*100, 3))
                ], axis=1).rename({
                    0: 'probabilité'
                }, axis=1)
                probas.sort_values(by='probabilité', ascending=False, inplace=True)
                probas['probabilité'] = probas['probabilité'].apply(lambda x: round(x, 2))
                st.write("Les classes prédites par le classifieur (top 5):")
                st.write(probas.head(5))
                st.bar_chart(data=probas, x='prdlabelcode', y="probabilité")

            proba_texte = logistic_regression_model.predict_proba(vector.reshape(1, -1))
            display_top_proba(proba_texte)

            st.markdown("<p class=\"page_title_3\">Classification de l'image (Réseau de neurones à convolutions)</p>", unsafe_allow_html=True)

            image = get_image(row.image)
            proba_image = vgg16_cnn_model.predict(image.reshape((1,224,224,3)))
            display_top_proba(proba_image)

            st.markdown("<p class=\"page_title_3\">Classification multimodale</p>", unsafe_allow_html=True)
            weight_texte = 0.8
            weight_image = 0.5
            proba_unifie = ((weight_texte * proba_texte + weight_image * proba_image)/(weight_image + weight_texte))
            display_top_proba(proba_unifie)    
 
#########
# Bilan #
#########

if choose == "Bilan":
    st.markdown('<p class="page_title_1">Bilan</p>', unsafe_allow_html=True)
    
    st.write("""
    Ce projet nous a permis de nous confronter aux différentes étapes d'un projet date science de l'exploration des données, à la phase de preprocessing et la modélisation.
    Le sujet était un challenge car nous il necessitant la mobilisation de plusieurs notions vues lors du parcours : langage python, analyse de données textuelles, analyse des données images.
    Notre modèle final a une bonne capacité de prédictions sur certaines catégories (liste ?). Cependant il présente encore quelques faiblesses pour d'autres catégories de produits.
    Liste les perspectives :
    - Perspective 1
    - Perspective 2
    - Perspective 3
    """)    



# Livres, mangas, romans:               Lots de Livres et de Revues
# Livres2:                              Livres > ebooks
# Jeux videos:                          Jeux vidéo > Version physique
# Jeux videos:                          Jeux video > Version dématérialisee
# Figurines1:                           Figurines
# Figurines2:                           Jeux de rôle et jeux de figurines
# Accessoire jeux                      Accessoires jeux videos


