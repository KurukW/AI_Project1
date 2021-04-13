Suite à l'appel du 26 Mars:
Nous abandonnons l'idée du squelette parce que Mme Slama nous a dit de ne pas le faire ainsi.
Nous allons trouver un article qui explique une méthode d'analyse d'image et de main afin
de s'inspirer de celui-ci.



Notes pour nous-même:
Reconnaissance de signes dynamique:

pointer un endroit
ouvrir la main
fermer la main
mouvement de la main avec des doigts sorti



Notre ancienne procédure:
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





Maj 13/4
Problèmes:
- Prise de vidéo imprécise, le nombre d'images n'est pas fixe.
->Modifier la vidéo pour avoir le bon nombre d'image ou alors modifier le programme.
->Modifier programme pour  ne pas devoir le modifier à chaque fois
- Le modele, est-il bon?
- Comment capturer les données et les envoyer au modèle?
->Prendre deux secondes toutes les deux secondes? Le problème est si le mouvement se fait entre deux secondes..
->Enregistrer uniquement lorsqu'il y a du mouvement mais est-ce qu'on va vraiment détecter le mouvement?
->Enregistrer s'il y a du mouvement dans une zone et enregistrer slmt cette zone. C'est un peu bourrin mais ça peut fonctionner
- Il faut des vidéos. Absolument.
