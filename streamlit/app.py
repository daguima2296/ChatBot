# Stream Lit application
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_chat import message
import spacy
import emoji
import re
from sklearn.cluster import KMeans
from datetime import datetime
import string
import MLPpretrait as MP
from keras.models import load_model
st.set_page_config(
    page_title='Supply Chain - Satisfaction des clients',
    page_icon='bot.png',
    layout='wide',
    menu_items={'About': "# Cette application et le fruit de nombreux efforts "})
st.markdown(
    """
    <style>
        /* Style pour centrer le texte */
        .center {
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }
       .block-container{
            width:80%;
            margin-left: 7.4%;
            margin-right: 7.4%;
            padding:5%;
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="center">Supply Chain - ChatBot</h1>',unsafe_allow_html=True)

stop_words = ['au','aux','avec','ce','ces','dans','de','des','du','elle','en','et',
              'eux','il','ils','je','la','le','les','leur','lui','ma','mais','me',
              'même','mes','moi','mon','ne','nos','notre','nous','on','ou','par',
              'pas','pour','qu','que','qui','sa','se','ses','son','sur','ta','te',
              'tes','toi','ton','tu','un','une','vos','votre','vous','c','d','j',
              'l','à','m','n','s','t','y','été','étée','étées','étés','étant',
              'étante','étants','étantes','suis','es','est','sommes','êtes','sont',
              'serai','seras','sera','serons','serez','seront','serais','serait',
              'serions','seriez','seraient','étais','était','étions','étiez','étaient',
              'fus','fut','fûmes','fûtes','furent','sois','soit','soyons','soyez',
              'soient','fusse','fusses','fût','fussions','fussiez','fussent','ayant',
              'ayante','ayantes','ayants','eu','eue','eues','eus','ai','as','avons',
              'avez','ont','aurai','auras','aura','aurons','aurez','auront','aurais',
              'aurait','aurions','auriez','auraient','avais','avait','avions','aviez',
              'avaient','eut','eûmes','eûtes','eurent','aie','aies','ait','ayons','ayez',
              'aient','eusse','eusses','eût','eussions','eussiez','eussent',"a", "car",
              "c'est", "j'ai", "amazon", "alors", "tout", "si", "cet", "chez", "donc", 
              "plus", "aussi", "d'un", "très", "tres","rien","sans","qu'il","encore",
              "ça","cette","autre","non","après","ni","jamais","toujours","fois",
              "maintenant","leur","comme","depuis","fait","quand","dit","aucun",
              "cela","plusieurs","moins","peut","cas","part","faut","tous","aucune",
              "vraiment","là","deux","avant","trouve","leurs","peu","souvent","vais",
              "où","beaucoup","passe","passé","quelques","va","disant","sous","trop",
              "ans","quoi","chaque","peux","déjà","surtout","suite","autres","font",
              "toute","vu","puis","Bonjour","coup","seul","parfois""fais","malgré",
              "entre","mal","ailleurs","jusqu","dernière","pris","pense","dont","bout",
              "dois","mis","sais","pu","gros","toutes","dois","moment","simplement",
              "reste","monde","lors","tant","bref","pendant","aller","fuir","veut",
              "certains","lorsque","dernier","disent","via","lieu","pourtant","gens",
              "sauf","vers","etc","fin","parce","merci","grand","comment","an","merci",
              "vite","port","assez","peuvent","seulement","sûr","pourquoi","viens",
              "seule","fini","final","auprès","faite","dès","tel","plutôt","bref",
              "afin","presque","faire","être","avoir","rakuten","cdiscount","wish",
              "lorsqu",'apres', 'aupres','bonjour', 'ca', 'deja', 'derniere', 'etaient',
              'etais', 'etait', 'etant', 'etante','etantes', 'etants', 'ete', 'etee',
              'etees', 'etes', 'etiez', 'etions', 'etre','eumes', 'eutes', 'fumes',
              'futes', 'malgre', 'meme', 'plutot',"ratuken","aujourd","hui",
              "aujourd'hui","le","la","les","fu"]
nlp = spacy.load('fr_core_news_sm')
def contient_chiffre(texte):
    for char in texte:
        if char.isdigit():
            return True
    return False

def lemmatisation_etc(text, mots_exclus=stop_words):
    doc = nlp(text.lower())
    lemmatized_tokens = []
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not contient_chiffre(str(token)):
                    if token.text not in mots_exclus:
                        if token.lemma_ not in mots_exclus:
                            lemmatized_tokens.append(token.lemma_)
    return lemmatized_tokens
def extract_punctuation(s):
    return ''.join([c for c in s if c in string.punctuation])

ner = nlp.get_pipe("ner")
entity_labels = ner.labels
def text_ner(text):
    dict = {i: 0 for i in entity_labels}#on initialise un dictionnaire à 0 pour l'ensemble des ner
    doc=nlp(text)#on creer l'objet nlp par rapport au text d'entrer
    for ent in doc.ents:#on parcours les entité qu'il y a dans le texte
        dict[ent.label_]+=1#on fait plus 1 pour chaque label qu'on trouve
    for key,value in dict.items():#on parcours le dictionnaire
        dict[key]=round(100*(value/len(doc)),2)#on fait le pourcentage par rapport au nombre de mots dans le texte
    return dict

pos_labels=["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X","SPACE"]
def text_pos(text):
    dict = {i: 0 for i in pos_labels}
    doc=nlp(text)
    for token in doc:
        dict[token.pos_]+=1
    for i in dict:
        dict[i]=round(100*(dict[i]/len(doc)),2)
    return dict

def preparecomments(txt):
    expression_signs = ['!', '?']
    txt = emoji.demojize(txt)
    txt = re.sub(r':[a-zA-Z0-9_]+:', '', txt)
    tabpunc=extract_punctuation(txt)
    expression_pattern = '|'.join([re.escape(s) for s in expression_signs])
    dayto=datetime.today().strftime('%A')
    nerdt=text_ner(txt)
    posdt=text_pos(txt)
    data={"nb_mots_comm":[len(txt.split())],
         "expression_count_comm":[txt.count(tabpunc)],
         "annee_comm":[datetime.today().strftime('%Y')],
         "Contains_Ellipsis_comm":[1 if '...' in txt else 0],
         "Monday":[1 if dayto == "Monday" else 0],"Tuesday":[1 if dayto == "Tuesday" else 0],
         "Wednesday":[1 if dayto == "Wednesday" else 0],"Thursday":[1 if dayto == "Thursday" else 0],
         "Friday":[1 if dayto == "Friday" else 0],"Saturday":[1 if dayto == "Saturday" else 0],
         "commentaireAmeliore":[' '.join(lemmatisation_etc(txt))],
         "LOC_count":[nerdt["LOC"]],"MISC_count":[nerdt["MISC"]],"ORG_count":[nerdt["ORG"]],"PER_count":[nerdt["PER"]],
         "ADJ_count":[posdt["ADJ"]],"ADP_count":[posdt["ADP"]],"ADV_count":[posdt["ADV"]],"AUX_count":[posdt["AUX"]],
         "CCONJ_count":[posdt["CCONJ"]],"DET_count":[posdt["DET"]],"INTJ_count":[posdt["INTJ"]],"NOUN_count":[posdt["NOUN"]],
         "NUM_count":[posdt["NUM"]],"PART_count":[posdt["PART"]],"PRON_count":[posdt["PRON"]],"PROPN_count":[posdt["PROPN"]],
         "PUNCT_count":[posdt["PUNCT"]],"SCONJ_count":[posdt["SCONJ"]],"SYM_count":[posdt["SYM"]],"VERB_count":[posdt["VERB"]],
         "X_count":[posdt["X"]],"SPACE_count":[posdt["SPACE"]]
         }
    return pd.DataFrame(data)

RLsave=joblib.load("RL.sav")
TRLsave=joblib.load("TRL.sav")
TFKB=joblib.load("TFKG.sav")
KB=joblib.load("KG.sav")
TFKG=joblib.load("TFKGG.sav")
KG=joblib.load("KGG.sav")
RF=joblib.load("RF.sav")
preMLPsuitsave,preMLPsave=joblib.load("PMLP.sav")
MLPsave=load_model("my_model.h5")
# Fonction pour envoyer une requête à l'API d'analyse de sentiment
def analyze_sentiment(comment):
    Test=preparecomments(comment)
    prediction=RLsave.predict(TRLsave.transform(Test["commentaireAmeliore"]))
    predRL=RF.predict(Test.drop(columns=["commentaireAmeliore"]))
    predictionMLP=preMLPsuitsave.transform(preMLPsave.transform(Test))
    predictionMLP=MLPsave.predict(predictionMLP)
    if predictionMLP>0.5:
        predictionMLP=1
    else:
        predictionMLP=0
    prediction+=(predRL+predictionMLP)
    if prediction>2:
        prediction=1
        cluster=KG.predict(TFKG.transform(Test.commentaireAmeliore.values))
    else:
        prediction=0
        cluster=KB.predict(TFKB.transform(Test.commentaireAmeliore.values))
    return (prediction,cluster)

# Fonction pour obtenir le texte saisi par l'utilisateur
def get_text():
    input_text = st.text_input("Saisissez votre message ici: ", "", key="input")
    return input_text

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    user_input = get_text()   
    if user_input:
        # Analyser le sentiment du commentaire saisi par l'utilisateur
        sentiment,prob = analyze_sentiment(user_input)
        # Déterminer la réponse en fonction du sentiment
        if sentiment == 1:
            response="""Nous sommes ravis de savoir que vous êtes pleinement satisfait de nos produits et services. 
            Pour ne rien manquer de nos offres promotionnelles exclusives réservées à nos fidèles clients, nous vous invitons à nous suivre sur nos réseaux sociaux. Vous serez ainsi informé en avant-première de nos nouvelles promotions, réductions et événements spéciaux."""
        elif sentiment == 0:
            if prob==0:
                response= """Si nos services et nos produits ne vous procurent pas satisfaction, nous en sommes désolés et nous vous prions de nous en excuser.
                Vous avez la possibilité de réaliser une demande de remboursement via votre espace client dans la rubrique **Mes commandes**. 
                Soyez assuré que nous mettrons tout en œuvre pour répondre votre demande et vous satisfaire de la meilleure des manières."""
            if prob==1:
                response= "Nous sommes désolés pour la gêne occassionée, sachez que nous nous efforçons de livrer tous nos colis dans les meilleurs délais. Pour votre prochain achat vous pourrez choisir le point de livraison que vous souhaitez avec une livraison prioritaire assurée par Chronopost. Vous aurez également la possibilité de suivre la livraison de votre colis en temps réel."
            if prob==2:
                response= "Nous vous prions de bien vouloir nous excuser pour le désagrement lié au temps d'attente lors de votre appel téléphonique. Nous vous informons que vous pouvez nous contacter désormais par la messagerie de vos réseaux sociaux préférés (WhatsApp, Instagram, Facebook), un conseiller sera ravi de pouvoir répondre à vos questions dans les plus courts délais. Soyez assuré que nous mettons tout en oeuvre pour répondre dans les plus brefs délais."
            if prob==3:
                response= """Nous sommes désolés pour le désagrement, sachez que votre satisfaction est notre priorité ! En dédommagement, nous vous offrons un code de bon d'achat de 10€ valable un an sur tous les produits de notre site internet.
                Code à renseigner lors de votre prochain achat : SAV10"""
        else:
            response = "Je n'ai pas pu analyser le sentiment de votre commentaire."

        # Ajouter le texte saisi par l'utilisateur et la réponse générée aux listes dans l'état de la session
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
    # Afficher les messages générés et les messages de l'utilisateur
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))  # Afficher un message généré
            if i!=0:
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # Afficher un message de l'utilisateur

    # Afficher un message de bienvenue si la liste 'generated' est vide
    if not st.session_state['generated']:
        WelText="""Salut ! 
                👋Je suis Botty, votre conseiller virtuel préféré. Comment allez-vous aujourd'hui ? Nous apprécions énormément vos commentaires, ils nous aident à améliorer nos produits et services.
                Pourriez-vous répondre à quelques questions pour moi ? Comment évaluez-vous la qualité des produits que vous avez acheté ? Avez-vous rencontré des problèmes de livraison ou avec notre service client ? Si c'est le cas, pourriez-vous partager une expérience spécifique afin que nous puissions mieux comprendre comment améliorer nos services ?
                Merci beaucoup de prendre le temps de partager votre avis avec nous. Nous sommes impatients de lire vos commentaires !"""
        st.session_state.generated.append(WelText)
        st.session_state.past.append("")
        message(WelText, key='welcome')

footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    opacity: 0.5;
    width: 100%;
    height: 30px;
    background-color: #000000;
    color: #cbcbcb;
    text-align: center;
    z-index: 9999;
}

.separator {
    display: inline-block;
    margin: 0 5px;
    font-size: 20px;
    color: #cbcbcb;
}
.footer a {
    color:  #cbcbcb;
    text-decoration: none;
}
.footer a:hover {
    color:  #cbcbcb;
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>
    VerbaSatisPyzer
    <span class="separator">&#8226;</span>
    Analyse des Verbatims pour la Satisfaction Client
    <span class="separator">&#8226;</span>
    <a href="https://www.linkedin.com/in/daguima/">Daniela Guisao Marin</a>
    <span class="separator">&#8226;</span>
    <a href="https://formation.datascientest.com/data-scientist-landing-page?utm_term=data%20scientist%20formation%20continue&utm_campaign=%5Bsearch%5D+data+scientist&utm_source=adwords&utm_medium=ppc&hsa_acc=9618047041&hsa_cam=15509646166&hsa_grp=130979844436&hsa_ad=568081578908&hsa_src=g&hsa_tgt=kwd-314862478488&hsa_kw=data%20scientist%20formation%20continue&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gclid=CjwKCAjwm4ukBhAuEiwA0zQxk4kbRxVPy93ZRGk_bXloREijHxs7_bC2i3K8GeYkgO8vBdvDy-bh3hoC-40QAvD_BwE">
    Formation Continue Data Scientist</a> 
    <span class="separator">&#8226;</span>
    Promotion Septembre 2022
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
