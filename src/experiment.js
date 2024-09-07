import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import jsPsychHtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';

const jsPsych = initJsPsych();

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
    try {
      const response = await fetch('./bigram_pairs.csv');
      const csvText = await response.text();
      const rows = csvText.split('\n').map(row => row.trim()).filter(row => row);
      return rows.map(row => {
        const bigrams = row.split(',').map(bigram => bigram.trim());
        // Randomly shuffle the order of bigrams in each pair
        return jsPsych.randomization.shuffle(bigrams);
      });
    } catch (error) {
      console.error('Error loading bigram pairs:', error);
      return [];
    }
  }

  const bigramPairs = await loadBigramPairs();

  if (bigramPairs.length === 0) {
    console.error('No bigram pairs loaded. Ending experiment.');
    jsPsych.endExperiment('Error: No bigram pairs available.');
    return;
  }

  // Randomize the order of bigram pairs
  const randomizedBigramPairs = jsPsych.randomization.shuffle(bigramPairs);

  // Add this function to set global styles
  function setGlobalStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .jspsych-content {
        max-width: 90% !important;
        font-size: 24px !important;
      }
      .jspsych-btn {
        font-size: 20px !important;
        padding: 15px 25px !important;
        margin: 10px !important;
      }
      #experiment-timer {
        font-size: 24px !important;
      }
      .comfort-choice-button {
        font-size: 24px !important;
        padding: 20px 30px !important;
        margin: 15px !important;
        min-width: 200px !important;
      }
    `;
    document.head.appendChild(style);
  }

  // Call this function at the start of the experiment
  setGlobalStyles();

  // Set up the timeline
  const timeline = [];

  let experimentStartTime;
  let timerInterval;

  // Set a fixed time for the experiment (30 seconds)
  const maxTimeAllowed = 30; // Time in seconds

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

  // Start experiment screen
  const startExperiment = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<p style="font-size: 28px;">Press the button below to begin the experiment.</p>`,
    choices: ["Start"],
    button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
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
  };

  timeline.push(startExperiment);

  // Function to create a typing trial for each bigram
  function createTypingTrial(bigram, trialId) {
    return {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `<div class="jspsych-content-wrapper">
                   <div class="jspsych-content">
                     <p>Type <b>${bigram}</b> three times.</p>
                     <p id="user-input" style="font-size: 24px; min-height: 30px; letter-spacing: 2px;"></p>
                     <p id="feedback" style="text-align: center; min-height: 24px;"></p>
                     <p id="error-message" style="color: red; text-align: center; min-height: 24px;"></p>
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
        const userInputElement = document.querySelector('#user-input');
        const feedbackElement = document.querySelector('#feedback');
        const errorMessageElement = document.querySelector('#error-message');
  
        const keyData = [];
        let lastKeyDown = null;
  
        const handleKeydown = (event) => {
          if (trialEnded) return;
        
          const typedKey = event.key.toLowerCase();
          
          // Prevent default behavior for all keys except 'Backspace'
          if (typedKey !== 'backspace') {
            event.preventDefault();
          }
          
          const currentPosition = typedSequence.length % bigram.length;
          const correctKey = bigram[currentPosition];
        
          // Log the keydown event
          keyData.push({
            type: 'keydown',
            key: typedKey,
            time: performance.now(),
            correct: typedKey === correctKey
          });
          
          lastKeyDown = typedKey;
        
          // Handle backspace
          if (typedKey === 'backspace') {
            if (typedSequence.length > 0) {
              typedSequence = typedSequence.slice(0, -1);
              updateUserInputDisplay();
            }
            return;
          }
        
          // Handle typed key
          if (typedKey === correctKey) {
            typedSequence += typedKey;
            updateUserInputDisplay();
            errorMessageElement.innerHTML = "";
        
            if (typedSequence.length === bigram.length * 3) {
              correctCount++;
              feedbackElement.innerHTML = "Correct! Press any key to continue.";
              if (correctCount === 1) {
                endTrial();
              }
            }
          } else {
            errorMessageElement.innerHTML = `Incorrect. Try again.`;
          }
        };
        
        const handleKeyup = (event) => {
          if (trialEnded) return;
        
          const typedKey = event.key.toLowerCase();
        
          // Log keyup
          keyData.push({
            type: 'keyup',
            key: typedKey,
            time: performance.now()
          });
        };
          
        const updateUserInputDisplay = () => {
          let displayHTML = '';
          for (let i = 0; i < typedSequence.length; i++) {
            const charClass = typedSequence[i] === bigram[i % bigram.length] ? 'correct' : 'incorrect';
            displayHTML += `<span class="${charClass}">${typedSequence[i]}</span>`;
          }
          userInputElement.innerHTML = displayHTML;
        };
  
        const endTrial = () => {
          if (trialEnded) return;
          trialEnded = true;
          document.removeEventListener('keydown', handleKeydown);
          document.removeEventListener('keyup', handleKeyup);
          jsPsych.finishTrial({
            correctCount: correctCount,
            keyData: keyData
          });
        };
  
        document.addEventListener('keydown', handleKeydown);
        document.addEventListener('keyup', handleKeyup);
  
        // Add styles for correct and incorrect characters
        const style = document.createElement('style');
        style.textContent = `
          #user-input .correct { color: green; }
          #user-input .incorrect { color: red; }
        `;
        document.head.appendChild(style);
      }
    };
  }
  
  // Create trials for each pair of bigrams
  randomizedBigramPairs.forEach(([bigram1, bigram2], index) => {
    timeline.push(createTypingTrial(bigram1, `trial-${index+1}-1`));
    timeline.push(createTypingTrial(bigram2, `trial-${index+1}-2`));
  
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `<div class="jspsych-content-wrapper">
                   <div class="jspsych-content">
                     <p style="font-size: 28px;">Which pair was easier (more comfortable) to type?</p>
                   </div>
                 </div>`,
      choices: [bigram1, bigram2, "No difference"],
      button_html: '<button class="jspsych-btn comfort-choice-button">%choice%</button>',
      data: {
        task: 'comfort_choice',
        bigram1: bigram1,
        bigram2: bigram2
      },
      on_finish: function(data) {
        console.log('Comfort choice data:', data);  // For debugging
        if (data.response !== null && data.response !== undefined) {
          if (data.response === 0) {
            data.comfortable_pair = bigram1;
          } else if (data.response === 1) {
            data.comfortable_pair = bigram2;
          } else {
            data.comfortable_pair = "no difference";
          }
        } else {
          console.log('No response recorded for comfort choice.');
          data.comfortable_pair = null;
        }
      }
    });
  });

  // Thank you screen and end experiment
  const thankYouTrial = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p style="font-size: 28px;">Thank you for participating! Press the button to finish.</p>
                 </div>
               </div>`,
    choices: ["Finish"],
    button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
    on_finish: function() {
      endExperiment("Experiment completed. Thank you for your participation!");
    }
  };

  timeline.push(thankYouTrial);

  // Function to store data on OSF
  async function storeDataOnOSF(data) {
    if (!osfToken) {
      console.error('Error: OSF API token not available. Data will not be stored on OSF.');
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
        const errorDetails = await uploadResponse.json();
        console.error(`Upload error! Status: ${uploadResponse.status}, Details: ${errorDetails.message}`);
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