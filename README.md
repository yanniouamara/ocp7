## Introduction

# Implémenter un modèle de Scoring

Evaluer la solvabilité de client auprès d’une banque, en vue d’un prêt.

Pour ceci, nous avons à disposition plusieurs tableau d’informations. Ceux-ci contiennent les informations personnelles :
-	Nombre d’enfant
-	Type d’emploi occupé
-	Situation maritale, etc

Et de données bancaires :
-	Dernières dates d’emprunts
-	Salaires, etc
-	
Les contraintes sont d’obtenir une classification entre les non solvables et les clients solvables, par entraînement supervisée.

Ainsi nous avons un tableau présentant les données de clients anonymes, ayant été solvable ou non.

## Méthodologie d’entraînement du modèle

# Nettoyage des données

L’exploration et le nettoyage des données s’est effectué en deux itérations.

# 1ere iteration

On a tout d’abord recherché les valeurs manquantes dans notre jeu de données application_train.csv

 ![image](https://user-images.githubusercontent.com/40555695/156807786-9026446c-5224-425d-82d8-f47e3cb76f3b.png)


Différent choix d’imputation de valeurs manquantes, à la moyenne, par les voisins les plus proches, …
On a choisi ici l’algorithme MissForest. L’algorithme permet de compléter les données manquantes, numériques ou catégorielles.
Il utilise pour cela un algorithme Tree Based qui permet l’imputation selon les échantillons les plus proches, par moyenne et autres variables statistiques.
S’en suit une gestion des valeurs aberrantes. Or il s’avère qu’il y a des valeurs aberrantes pour la colonne « DAYS_EMPLOYED » d’application_train et test. Cette valeur spécifique ne signifie rien sur la solvabilité du client, ainsi elle a pu se propager.

![image](https://user-images.githubusercontent.com/40555695/156807869-819afc5f-1c25-435b-9dec-c4ffa2d345a3.png)

# 2eme iteration

Gestion des valeurs aberrantes puis imputation par MissForest des valeurs manquantes. Les données catégorielles ont auparavant été mise sous forme de one-hot-encoding.
Les features choisies ont été déterminées par RandomForest, les 40 premières features ont été conservées.


# Modèle choisi

Le modèle choisi doit correspondre aux contraintes de l’énoncé. Il doit donc s’agir d’un classifieur binaire permettent une probabilité d’appartenance à chaque classe. De plus, une explication d’attribution à une classe ou une autre doit être possible.
LightGBM est l’algorithme qui sera utilisé. Il permet de choisir un classifieur binaire. Il permet aussi de choisir la métrique d’entrainement ici, ce qui est très important au vu du sujet choisi.

# Spécificités du modèle
#Fonction coût métier
Les modèles vont être évalués selon une métrique adaptée au problème métier. On part du principe que la société financière veut optimiser ses gains (et donc minimiser ses pertes).
 
La société étant un organisme de prêts, outre la bonne détection des prêts accordés ou refusés correctement (Vrais Négatifs ou Vrais Positifs), il va falloir aussi minimiser autant que possible les cas où un emprunteur serait identifié comme solvable alors qu'il ne l'est pas (Faux Négatifs). Il est moins important de se concentrer sur les emprunteurs identifiés comme insolvables alors qu'ils le sont (Faux Positifs), même si cela a aussi un coût pour l’organisme financier (d'opportunité, en premier lieu, mais aussi éventuellement de réputation).
En effet, le parti pris dans mon cas est de minimiser les Faux Positifs. On veut éviter au maximum de proposer un prêt à une personne potentiellement non solvable. On utilisera donc la métrique Area Under the ROC curve, donc « AUC » de notre classifieur.
Enfin, le partage des classes est très désavantagé. On va donc aussi changer les poids d’entrainement de notre algorithme en choisissant le paramètre « is_unbalanced=True ».

![image](https://user-images.githubusercontent.com/40555695/156807957-0c089906-0641-4829-a26d-6f900716bc5d.png)


# Algorithme d'optimisation
Afin de déterminer les paramètres de notre modèle, nous utilons le BayesSearchCv de la librairie scikit-optimize. Il nous permet d’explorer un espace de paramètre de manière non exhaustive en utilisant une fonction de score spéficique.
Pour cela, il va inférer à partir d’ancien score obtenu par notre fonction de coût ici (metric = « AUC »). Il va donc so’rienter petit à petit vers un minimum local voir global.


#Métrique d’évaluation

![image](https://user-images.githubusercontent.com/40555695/156808090-8715bfe3-37c3-495e-bd61-e2c5683b09ef.png)

La métrique d’évaluation est finalement ce qui va nous permettre d’évaluer si notre modèle permet d’atteindre notre objectif de minimisation des FPs. 
Ainsi nous utilisons le ratio entre les Faux Positifs et les Vrais Positifs afin de trouver un compromis entre le fait de reconnaître la solvabilité de quelqu’un, et d’éviter de prêter à un client.

True positive Rate: 		![image](https://user-images.githubusercontent.com/40555695/156807992-224294c6-637e-44b6-8d2f-9bcb5dec5669.png)	      

False Positive Rate: ![image](https://user-images.githubusercontent.com/40555695/156808025-6706edb0-aeda-4132-a865-1b01b11d5eda.png)


En sorti de notre algorithme, nous utilisons le predict_proba de notre modèle.
On pbtient donc une probailité d’attribution à une classe ou une autre. On peut donc utiliser cette probabilité et évaluer le FPR et TPR selon différent seuil de probabilité.
On se retrouve donc avec ces deux graphiques :
![image](https://user-images.githubusercontent.com/40555695/156808308-b75abc1e-a804-42f6-9937-e85361e82773.png) ![image](https://user-images.githubusercontent.com/40555695/156808332-b34565bc-f1f7-4d5a-ab3c-6d09de402983.png)
![image](https://user-images.githubusercontent.com/40555695/156808346-1fe1c934-eb0f-4dc9-b1fe-2d044196036f.png)


On peut donc choisir notre seuil selon ces graphiques, qui est à 0.9. La matrice de confusion associée est très correcte, seulement 5 faux-positifs sur un ensemble de 200 échantillons, mais représentant tout de même 40% des Vrai non solvable totaux.

# L’interprétabilité globale et locale du modèle

Le modèle permet de sortir plusieurs paramètres.
Celui qui nos intéresse globalement est l’importance des features globales dans notre modèle. On peut donc obtenir l’importance de nos différentes features pour donner suite à l’entraînement de notre modèle, et donc voir l’influence possible que peut avoir une donnée d’un client sur la probabilité de sa solvabilité.
De manière locale, il est possible d’utiliser l’algorithme SHAP qui permet de sortir de manière précise les raisons d’une classification.
L’algorithme SHAP permet de sortir d’un modèle Tree l’importance des caractéristiques d’entraînement pour chaque échantillon séparément. Ainsi on peut obtenir des graphes de ce type :

 ![image](https://user-images.githubusercontent.com/40555695/156808405-b0628275-aa81-4854-921a-829051c9877e.png)

Ces importances sont négatives ou positives selon la nature de l’impact sur la sortie de notre modèle.

#Limites et Améliorations du modèle

Manque de données d’entrées
-	Le modèle actuel possède peu de données, nous avons seulement 1050 échantillons.

Objectif interne à l’entreprise
-	Parmi ceux non solvable réel, presque 50% d’entre eux sont prédit comme solvable. Il s’agit d’un compromis où beaucoup prêts seraient perdus si le seuil était diminué. Il s’agit donc d’une considération en dehors de ce projet.

Domain Knowledge
-	Une meilleure connaissance du domaine permettrait une utilisation des autres tableaux à disposition. En effet, augmenter le nombre de dimension permettrait d’effectuer du features  engineering afin de réduire la dimension de nos données.
