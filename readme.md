# bigram-typing-comfort-experiment
# heavily modified from jspsych-typing

## Build 

If you'd like to run the experiment locally (or remotely with vite):

1. Clone the project's repository (`git clone git@github.com:binarybottle/bigram-typing-comfort-experiment.git`)
2. `cd bigram-typing-comfort-experiment`
3. Make sure you have `Node.js >= 18.x`
4. Run `npm install` to install the necessary dependencies
5. `npm install vite`
6. `npm install jspsych @jspsych/plugin-html-button-response @jspsych/plugin-html-keyboard-response @jspsych/plugin-survey-multi-choice`
7. Run `npm run dev` to start the preview server
8. Visit `localhost:8080` in your browser to preview the project.

Remote production:
7. Add ``base: 'typing/bigram-comfort-study/',`` to vite.config.js
8. Run `npm run build` to generate static files for use in deployment on platforms such as GITHUB.
9. mkdir /home/binarybottle/binarybottle.com/bigram-comfort-study
10. cp -R dist/* /home/binarybottle/binarybottle.com/bigram-comfort-study/

