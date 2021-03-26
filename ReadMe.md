Ceci est le readme

Movement_detection.py : Détection du mouvement







Notes pour nous-même:
Reconnaissance de signes dynamique:

pointer un endroit
ouvrir la main
fermer la main
mouvement de la main avec des doigts sorti



Il faut demander à Mme Slama sur comment analyser les vidéos.
Le problème des vidéos est qu'elles n'ont un sens qu'ensemble.
On ne sait pas comment les analyser...
-> Nico s'occupe de faire des recherches là dessus

Notre procédure:
Isoler les mains pour construire le squelette.
Replacer le squelette dans l'image et utiliser le DL pour analyser le mouvement.
et la disposition du squelette.




Interaction avec HMI
- Série temporelle : LSTM
- Concaténer les images ensemble pour en avoir une seule
- En channel ça semble bien pour analyser les vidéos
- CVV Pour reconnaissance de main

Travailler avec l'image, c'est plus simple pour nous sachant qu'on fabrique
déjà notre Squelette.
