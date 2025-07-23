# bigram-typing-comfort-experiment
Arno Klein: binarybottle GitHub name 
bigram-typing-comfort-experiment GitHub repo
(heavily modified from jspsych-typing repo)

## README

The code takes in a list of bigram pairs and presents, and for each pair,
presents each bigram in a web browser and instructs the user to type the 
bigram three times in a row. Once both bigrams are typed three times in a row
correctly, the user is asked to choose which is easier (more comfortable) to type.
All timing and selection data are saved to OSF.io and return codes sent to Prolific.
er
## Build 

Installation steps for either local/remote hosting:

1. Clone the project's repository (``git clone git@github.com:binarybottle/bigram-typing-preference-study.git``)
2. ``cd bigram-typing-comfort-experiment``
3. Make sure you have ``Node.js >= 18.x``
4. Run ``npm install`` to install the necessary dependencies
5. ``npm install vite``
6. ``npm install jspsych @jspsych/plugin-html-button-response @jspsych/plugin-html-keyboard-response @jspsych/plugin-survey-multi-choice``

Run the experiment locally:

7. Run ``npm run dev`` to start the preview server
8. Visit ``localhost:8080`` in your browser to preview the project.

Remote production (with vite):

7. Add ``base: '/typing/bigram-prolific-study/',`` to vite.config.js
8. ``export STUDY='/home/binarybottle/arnoklein.info/typing/bigram-prolific-study'``
8. ``mkdir $STUDY``
9. ``cp ./token.json $STUDY/``
10. ``cp -R bigram_tables $STUDY/``
11. ``npm run build``
12. ``cp -R dist/* $STUDY/``
