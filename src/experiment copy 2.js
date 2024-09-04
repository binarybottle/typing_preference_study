import "jspsych/css/jspsych.css";
import "./style.css";
import jsPsych from "./prepare";
import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";
import jsPsychHtmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";

async function runExperiment() {
  // Load participant data
  async function loadParticipantData() {
    const participant_id = jsPsych.data.getURLVariable("PROLIFIC_PID") || "1"; // Default to participant 1 for testing
    const response = await fetch(`./sublists/participant_${participant_id}.json`);
    const data = await response.json();
    return data;
  }

  const sublist = await loadParticipantData();

  // Set up an empty timeline
  const timeline = [];

  // Don't show the default progress bar
  jsPsych.opts.show_progress_bar = false;

  let experimentStartTime;
  let timerInterval;

  // Retrieve max_time_allowed from jsPsych settings
  const maxTimeAllowed = jsPsych.getInitSettings().max_time_allowed || 30; // Default to 30 seconds if not set

  // Function to format time as mm:ss
  function formatTime(milliseconds) {
    const totalSeconds = Math.ceil(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  // Function to create and update the timer display
  function createAndUpdateTimer(timeRemaining) {
    let timerElement = document.getElementById('experiment-timer');
    if (!timerElement) {
      timerElement = document.createElement('div');
      timerElement.id = 'experiment-timer';
      timerElement.style.position = 'fixed';
      timerElement.style.top = '10px';
      timerElement.style.right = '10px';
      timerElement.style.padding = '10px';
      timerElement.style.background = '#f0f0f0';
      timerElement.style.border = '1px solid #ddd';
      timerElement.style.borderRadius = '5px';
      timerElement.style.fontSize = '18px';
      document.body.appendChild(timerElement);
    }
    timerElement.textContent = `Time remaining: ${formatTime(timeRemaining)}`;
  }

  // Add global timer to end experiment after max time
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `<p>Press any button to begin.</p>`,
    choices: ["Start"],
    on_finish: () => {
      experimentStartTime = performance.now();
      const maxTime = maxTimeAllowed * 1000; // Convert to milliseconds

      // Start updating the timer
      timerInterval = setInterval(() => {
        const elapsedTime = performance.now() - experimentStartTime;
        const remainingTime = Math.max(0, maxTime - elapsedTime);

        createAndUpdateTimer(remainingTime);

        if (remainingTime <= 0) {
          clearInterval(timerInterval);
          jsPsych.endExperiment("The maximum time allowed has been reached.");
        }
      }, 1000); // Update every second
    }
  });

  // Function to create a typing trial for each pair
  function createTypingTrial(pair) {
    const pairString = pair.join('').toLowerCase();
    return {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `<p>Type <b>${pairString}</b> three times.</p><p id="feedback" style="text-align: center;"></p><p id="error-message" style="color: red; text-align: center;"></p>`,
      choices: "ALL_KEYS",
      trial_duration: null, // No time limit per trial, but max time applies globally
      response_ends_trial: false, // The trial should not end automatically
      on_load: () => {
        let typedSequence = "";
        let correctCount = 0;
        const feedbackElement = document.querySelector('#feedback');
        const errorMessageElement = document.querySelector('#error-message');

        const handleKeydown = (event) => {
          const typedKey = event.key.toLowerCase();
          const currentPosition = typedSequence.length % pairString.length;

          // Check if the typed key is correct
          const correctKey = pairString[currentPosition];

          if (typedKey === correctKey) {
            typedSequence += typedKey;
            feedbackElement.innerHTML += `<span style="color: green;">${typedKey}</span>`;
            errorMessageElement.innerHTML = ""; // Clear error message

            // Check if the full pair has been typed
            if (typedSequence.length === pairString.length) {
              correctCount++;
              typedSequence = ""; // Reset for the next repetition
              if (correctCount < 3) {
                feedbackElement.innerHTML += `<br />`; // Prepare for next repetition
              }
            }
          } else {
            // Reset the sequence if incorrect
            typedSequence = "";
            correctCount = 0;
            feedbackElement.innerHTML = ""; // Clear feedback
            errorMessageElement.innerHTML = `Try again:`; // Show error message
          }

          // Finish the trial only when the pair is correctly typed three times in a row
          if (correctCount === 3) {
            document.removeEventListener('keydown', handleKeydown);
            jsPsych.finishTrial({
              correctSequence: pairString,
              correctCount: correctCount,
            });
          }
        };

        // Add the event listener
        document.addEventListener('keydown', handleKeydown);
      },
      on_finish: () => {
        // Cleanup any listeners if needed (though this should be handled)
      }
    };
  }

  // Create trials for each pair in the sublist
  sublist.forEach(([pair1, pair2]) => {
    // Typing trials
    timeline.push(createTypingTrial(pair1));
    timeline.push(createTypingTrial(pair2));

    // Ask which pair was more comfortable
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `<p>Which pair was more comfortable to type?</p>`,
      choices: [pair1.join('').toLowerCase(), pair2.join('').toLowerCase()],
      on_finish: (data) => {
        jsPsych.data.write({
          comfortable_pair: data.response,
        });
      }
    });
  });

  // Thank you screen
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: '<p>Thank you for participating! Press any button to finish.</p>',
    choices: ["Finish"],
  });

  // Run the timeline
  jsPsych.run(timeline);

  // Clean up the interval and remove the timer when the experiment ends
  jsPsych.on('finish', () => {
    if (timerInterval) {
      clearInterval(timerInterval);
    }
    const timerElement = document.getElementById('experiment-timer');
    if (timerElement) {
      timerElement.remove();
    }
  });
}

// Start the experiment
runExperiment();