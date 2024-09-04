import "jspsych/css/jspsych.css";
import "./style.css";
import jsPsych from "./prepare";
import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";
import jsPsychHtmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";

async function runExperiment() {
  // ... (previous code remains the same)

  // Set a fixed time for the experiment (30 seconds)
  const maxTimeAllowed = 30; // Time in seconds

  // ... (timer-related functions remain the same)

  // Function to create a typing trial for each pair
  function createTypingTrial(bigram, trialId) {
    return {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `<div class="jspsych-content-wrapper">
                   <div class="jspsych-content">
                     <p>Type <b>${bigram}</b> three times.</p>
                     <p id="feedback" style="text-align: center;"></p>
                     <p id="error-message" style="color: red; text-align: center;"></p>
                   </div>
                 </div>`,
      choices: "ALL_KEYS",
      trial_duration: null,
      response_ends_trial: false,
      data: {
        trialId: trialId,
        correctSequence: bigram,
      },
      on_load: function() {
        let typedSequence = "";
        let correctCount = 0;
        let trialEnded = false;
        const feedbackElement = document.querySelector('#feedback');
        const errorMessageElement = document.querySelector('#error-message');
  
        const keyData = [];
        let lastKeyDown = null;
  
        const handleKeydown = (event) => {
          if (trialEnded) return;
          const typedKey = event.key.toLowerCase();
          const currentPosition = typedSequence.length % bigram.length;
          const correctKey = bigram[currentPosition];
  
          // If there's a previous keydown without a corresponding keyup, log it now
          if (lastKeyDown) {
            keyData.push({
              type: 'keyup',
              key: lastKeyDown,
              time: performance.now() - 1 // 1ms before current keydown
            });
          }
  
          keyData.push({
            type: 'keydown',
            key: typedKey,
            time: performance.now(),
            correct: typedKey === correctKey
          });
  
          lastKeyDown = typedKey;
  
          if (typedKey === correctKey) {
            typedSequence += typedKey;
            feedbackElement.innerHTML += `<span style="color: green;">${typedKey}</span>`;
            errorMessageElement.innerHTML = "";
  
            if (typedSequence.length === bigram.length) {
              correctCount++;
              typedSequence = "";
              if (correctCount < 3) {
                feedbackElement.innerHTML += `<br />`;
              }
            }
          } else {
            typedSequence = "";
            correctCount = 0;
            feedbackElement.innerHTML = "";
            errorMessageElement.innerHTML = `Try again:`;
          }
  
          if (correctCount === 3) {
            keyData.push({ type: 'success', time: performance.now() });
            endTrial();
          }
        };
  
        const handleKeyup = (event) => {
          if (trialEnded) return;
          const typedKey = event.key.toLowerCase();
          keyData.push({
            type: 'keyup',
            key: typedKey,
            time: performance.now()
          });
          if (typedKey === lastKeyDown) {
            lastKeyDown = null;
          }
        };
  
        const endTrial = () => {
          if (trialEnded) return;
          trialEnded = true;
          // If there's a final keydown without a keyup, log it now
          if (lastKeyDown) {
            keyData.push({
              type: 'keyup',
              key: lastKeyDown,
              time: performance.now()
            });
          }
          document.removeEventListener('keydown', handleKeydown);
          document.removeEventListener('keyup', handleKeyup);
          jsPsych.finishTrial({
            correctCount: correctCount,
            keyData: keyData
          });
        };
  
        document.addEventListener('keydown', handleKeydown);
        document.addEventListener('keyup', handleKeyup);
      }
    };
  }
  
  // ... (rest of the code remains the same)
}

// Start the experiment
runExperiment();