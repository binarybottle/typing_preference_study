import "jspsych/css/jspsych.css";
import "./style.css";
import jsPsych from "./prepare";
import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";
import jsPsychHtmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";

async function runExperiment() {
  // Load OSF API token
  async function loadOSFToken() {
    try {
      const response = await fetch('./configs/token.json');
      const data = await response.json();
      return data.osf_api_token;
    } catch (error) {
      console.error('Error loading OSF token:', error);
      return null;
    }
  }

  const osfToken = await loadOSFToken();

  // Load bigram pairs data
  async function loadBigramPairs() {
    const response = await fetch('./bigram_pairs.csv');
    const csvText = await response.text();
    const rows = csvText.split('\n').map(row => row.trim()).filter(row => row);
    return rows.map(row => row.split(',').map(bigram => bigram.trim()));
  }

  const bigramPairs = await loadBigramPairs();

  // Set up the timeline
  const timeline = [];

  let experimentStartTime;
  let timerInterval;

  // Set a fixed time for the experiment (e.g., 10 minutes)
  const maxTimeAllowed = 10; // Time in seconds (600 seconds = 10 minutes)

  // Function to format time as mm:ss
  function formatTime(milliseconds) {
    if (isNaN(milliseconds)) {
      console.error('Invalid time value:', milliseconds);
      return '00:00';
    }
    const totalSeconds = Math.max(0, Math.ceil(milliseconds / 1000));
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
      timerElement.style.zIndex = '1000'; // Ensure timer is on top
      document.body.appendChild(timerElement);
    }
    timerElement.textContent = `Time remaining: ${formatTime(timeRemaining)}`;
  }

  // Function to end the experiment, clean up, and store data
  function endExperiment(message) {
    if (timerInterval) {
      clearInterval(timerInterval);
    }
    const timerElement = document.getElementById('experiment-timer');
    if (timerElement) {
      timerElement.remove();
    }

    // Get the experiment data and store it on OSF
    const experimentData = jsPsych.data.get().values();
    storeDataOnOSF(experimentData);

    // End the experiment
    jsPsych.endExperiment(message);
  }

  // Add global timer to end experiment after max time
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `<p>Press any button to begin.</p>`,
    choices: ["Start"],
    on_finish: () => {
      experimentStartTime = performance.now();

      const maxTime = maxTimeAllowed * 1000; // Convert to milliseconds

      timerInterval = setInterval(() => {
        const elapsedTime = performance.now() - experimentStartTime;
        const remainingTime = Math.max(0, maxTime - elapsedTime);

        createAndUpdateTimer(remainingTime);

        if (remainingTime <= 0) {
          endExperiment("The maximum time allowed has been reached.");
        }
      }, 100); // Update every 100ms for smoother countdown
    }
  });

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
      on_load: () => {
        let typedSequence = "";
        let correctCount = 0;
        let trialEnded = false;
        const feedbackElement = document.querySelector('#feedback');
        const errorMessageElement = document.querySelector('#error-message');
  
        const keyData = [];
  
        const handleKeydown = (event) => {
          if (trialEnded) return;
          const typedKey = event.key.toLowerCase();
          const currentPosition = typedSequence.length % bigram.length;
          const correctKey = bigram[currentPosition];
  
          keyData.push({
            type: 'keydown',
            key: typedKey,
            time: performance.now()
          });
  
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
        };
  
        const endTrial = () => {
          if (trialEnded) return;
          trialEnded = true;
          document.removeEventListener('keydown', handleKeydown);
          document.removeEventListener('keyup', handleKeyup);
          jsPsych.finishTrial({
            trialId: trialId,
            correctSequence: bigram,
            correctCount: correctCount,
            keyData: keyData
          });
        };
  
        document.addEventListener('keydown', handleKeydown);
        document.addEventListener('keyup', handleKeyup);
      },
      on_finish: (data) => {
        jsPsych.data.write({
          trialId: data.trialId,
          correctSequence: data.correctSequence,
          correctCount: data.correctCount,
          keyData: data.keyData
        });
      }
    };
  }
  
  // Create trials for each pair of bigrams
  bigramPairs.forEach(([bigram1, bigram2], index) => {
    timeline.push(createTypingTrial(bigram1.toLowerCase(), `trial-${index}-1`));
    timeline.push(createTypingTrial(bigram2.toLowerCase(), `trial-${index}-2`));
  
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `<div class="jspsych-content-wrapper">
                   <div class="jspsych-content">
                     <p>Which pair was more comfortable to type?</p>
                   </div>
                 </div>`,
      choices: [bigram1, bigram2],
      on_finish: (data) => {
        jsPsych.data.write({
          comfortable_pair: data.response,
        });
      }
    });
  });

  // Thank you screen and end experiment
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p>Thank you for participating! Press any button to finish.</p>
                 </div>
               </div>`,
    choices: ["Finish"],
    on_finish: () => {
      endExperiment("Experiment completed. Thank you for your participation!");
    }
  });

  // Function to store data on OSF
  async function storeDataOnOSF(data) {
    if (!osfToken) {
      console.error('OSF API token not available. Data will not be stored on OSF.');
      return;
    }

    const osfNodeId = "jf8sc";
    const createFileUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=experiment_data_${Date.now()}.json`;

    try {
      // Step 1: Upload the data to the OSF using PUT
      const uploadResponse = await fetch(createFileUrl, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${osfToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload error! status: ${uploadResponse.status}`);
      }

      console.log('Data successfully stored on OSF');
    } catch (error) {
      console.error('Error storing data on OSF:', error);
    }
  }

  // Run the timeline
  jsPsych.run(timeline);
}

// Start the experiment
runExperiment();