# bigram-typing-comfort-experiment
# heavily modified from jspsych-typing

## Build 

Installation steps for either local/remote hosting:

1. Clone the project's repository (``git clone git@github.com:binarybottle/bigram-typing-comfort-experiment.git``)
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
8. mkdir /home/binarybottle/binarybottle.com/bigram-prolific-study/
9. Copy bigram_[3,80]pairs.csv to bigram-prolific-study/configs/
9. Copy public/configs/token.json (OSF token) to bigram-prolific-study/configs/
10. Run ``npm run build`` to generate static files
11. ``cp -R dist/* /home/binarybottle/binarybottle.com/typing/bigram-prolific-study/``
