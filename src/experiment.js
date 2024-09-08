import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import jsPsychHtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';

// Add the unhandled rejection handler here
window.addEventListener('unhandledrejection', function(event) {
  console.error('Unhandled promise rejection:', event.reason);
});

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables for countdown timer and required correct repetitions
let timeLimit = 30;  // Default time limit of 30 seconds for the entire experiment
let requiredCorrectRepetitions = 3;  // Default requirement to type the bigram correctly 3 times

let experimentStartTime;
let timerInterval;

// Load OSF API token
async function loadOSFToken() {
  try {
    const response = await fetch('./configs/token.json');
    const data = await response.json();
    console.log('OSF API token loaded');  // Log the loaded token
    return data.osf_api_token;
  } catch (error) {
    console.error('Error loading OSF token:', error);
    return null;
  }
}

// Load bigram pairs from a CSV file or text source
async function loadBigramPairs() {
  try {
    const response = await fetch('./bigram_pairs.csv');  // Replace with the correct path to your file
    const csvText = await response.text();
    const rows = csvText.split('\n').map(row => row.trim()).filter(row => row);
    return rows.map(row => {
      const bigrams = row.split(',').map(bigram => bigram.trim());
      return jsPsych.randomization.shuffle(bigrams);  // Randomly shuffle bigrams using the jsPsych instance
    });
  } catch (error) {
    console.error('Error loading bigram pairs:', error);
    return [];
  }
}

// Add global styles for the experiment
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
    #timer {
      position: fixed;
      top: 10px;
      right: 20px;
      font-size: 20px;
      color: #000;
    }
  `;
  document.head.appendChild(style);
}

// Typing trial function
function createTypingTrial(bigram, bigramPair, trialId) {
  let keyData = [];  // Move keyData here so it is accessible across the entire trial lifecycle

  let handleKeyPress, handleKeyUp;  // Declare variables for event handlers

  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p>Type <b>${bigram}</b> ${requiredCorrectRepetitions} times in a row without mistakes.</p>
                   <p id="user-input" style="font-size: 24px; letter-spacing: 2px;"></p>
                   <p id="feedback" style="text-align: center;"></p>
                   <p id="error-message" style="color: red; text-align: center;"></p>
                 </div>
               </div>`,
    choices: "ALL_KEYS",
    response_ends_trial: false,
    data: {
      trialId: trialId,
      correctSequence: bigram,
      bigramPair: bigramPair.join(", "),
      keyData: []  // To store key events
    },
    on_load: function () {
      let typedSequence = "";
      let correctSequenceCount = 0;  // Track the correct repetitions
      let streakStarted = false;  // Track whether we are in a streak
      const trialStartTime = performance.now();

      const userInputElement = document.querySelector('#user-input');
      const feedbackElement = document.querySelector('#feedback');
      const errorMessageElement = document.querySelector('#error-message');

      handleKeyPress = (event) => {
        const typedKey = event.key.toLowerCase();
        const keydownTime = performance.now() - trialStartTime;
        const expectedKey = bigram[typedSequence.length % bigram.length];
        const isCorrectKey = typedKey === expectedKey;

        // Log keydown event with keyup event to be added later
        const keyLog = {
          trialId: trialId,
          bigramPair: bigramPair.join(", "),
          bigram: bigram,
          expectedKey: expectedKey,
          typedKey: typedKey,
          isCorrectKey: isCorrectKey,
          keydownTime: keydownTime.toFixed(2),
          keyupTime: "",
          inStreak: false  // Will be updated when streak is complete
        };

        keyData.push(keyLog);

        if (isCorrectKey) {
          typedSequence += typedKey;
          userInputElement.textContent = typedSequence;

          // If bigram has been typed correctly, increment the correct count
          if (typedSequence.length % bigram.length === 0) {
            correctSequenceCount++;
          }

          // Check if the correct sequence count reached the required streak count
          if (correctSequenceCount === requiredCorrectRepetitions) {
            streakStarted = true;
            keyData.forEach(key => key.inStreak = true);  // Mark all the keys in the streak
            feedbackElement.textContent = "Correct!";
            console.log("Correct sequence typed 3 times, ending trial...");
            jsPsych.finishTrial({ keyData: keyData });
          }

        } else {
          // If mistake, reset the sequence and streak
          errorMessageElement.textContent = "Mistake detected. Start over.";
          typedSequence = "";
          correctSequenceCount = 0;
          userInputElement.textContent = "";  // Reset typed sequence
        }
      };

      handleKeyUp = (event) => {
        const keyupTime = performance.now() - trialStartTime;
        const lastKey = keyData[keyData.length - 1];  // Update last key's keyupTime
        if (lastKey) {
          lastKey.keyupTime = keyupTime.toFixed(2);
        }
      };

      document.addEventListener('keydown', handleKeyPress);
      document.addEventListener('keyup', handleKeyUp);
    },
    on_finish: function (data) {
      // Remove event listeners when the trial ends
      document.removeEventListener('keydown', handleKeyPress);
      document.removeEventListener('keyup', handleKeyUp);

      // Store key data for the trial
      data.keyData = keyData;  // Now keyData is available here

      console.log(`Typing trial for ${bigram} finished...`);  // Add a log when the trial finishes
    }
  };
}

// Comfort choice trial
function createComfortChoiceTrial(bigram1, bigram2, trialIndex) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<p style="font-size: 28px;">Which pair was easier (more comfortable) to type?</p>`,
    choices: [bigram1, bigram2, "No difference"],
    button_html: '<button class="jspsych-btn comfort-choice-button">%choice%</button>',
    data: {
      task: 'comfort_choice',
      bigram1: bigram1,
      bigram2: bigram2,
      trialId: `trial-${trialIndex}-choice`
    },
    on_finish: function (data) {
      data.comfortable_pair = data.response !== null ? [bigram1, bigram2, "No difference"][data.response] : null;
    }
  };
}

// Function to escape commas and wrap fields in quotes if necessary
function escapeCSVField(field) {
  if (typeof field === 'string' && field.includes(',')) {
    return `"${field.replace(/"/g, '""')}"`;  // Escape double quotes by doubling them
  }
  return field;
}

// Function to convert data to CSV format
function convertToCSV(data) {
  const csvHeaders = ['trialId', 'bigramPair', 'bigram', 'expectedKey', 'typedKey', 'isCorrectKey', 'keydownTime', 'keyupTime', 'inStreak', 'chosenBigram'];
  let csvContent = csvHeaders.join(',') + '\n';

  data.forEach(trial => {
    if (trial.keyData) {
      trial.keyData.forEach(keyEvent => {
        const row = [
          escapeCSVField(keyEvent.trialId || ''),
          escapeCSVField(keyEvent.bigramPair || ''),
          escapeCSVField(keyEvent.bigram || ''),
          escapeCSVField(keyEvent.expectedKey || ''),
          escapeCSVField(keyEvent.typedKey || ''),
          keyEvent.isCorrectKey !== undefined ? keyEvent.isCorrectKey : '',
          keyEvent.keydownTime !== undefined ? keyEvent.keydownTime : '',
          keyEvent.keyupTime !== undefined ? keyEvent.keyupTime : '',
          keyEvent.inStreak !== undefined ? keyEvent.inStreak : '',
          escapeCSVField(trial.chosenBigram || '')
        ];
        csvContent += row.join(',') + '\n';
      });
    }
  });

  return csvContent;
}

// Function to store data on OSF
async function storeDataOnOSF(data, format = 'csv') {
  console.log("Received data for upload:", data);
  const osfToken = await loadOSFToken();
  console.log("Using OSF API token:", osfToken);

  if (!osfToken) {
    console.error('Error: OSF API token not available. Data will not be stored on OSF.');
    return;
  }

  const osfNodeId = "jf8sc";
  const fileExtension = format === 'csv' ? 'csv' : 'json';
  const createFileUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=experiment_data_${Date.now()}.${fileExtension}`;

  let fileData;
  let contentType;

  if (format === 'csv') {
    console.log("Converting data to CSV format...");
    fileData = convertToCSV(data);
    contentType = 'text/csv';
  } else {
    console.log("Converting data to JSON format...");
    fileData = JSON.stringify(data);
    contentType = 'application/json';
  }

  try {
    console.log(`Attempting to upload data to OSF. URL: ${createFileUrl}`);
    const uploadResponse = await fetch(createFileUrl, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${osfToken}`,
        'Content-Type': contentType,
      },
      body: fileData,
    });

    console.log("OSF Upload Response:", uploadResponse);

    if (!uploadResponse.ok) {
      const errorDetails = await uploadResponse.text();
      throw new Error(`Upload error! Status: ${uploadResponse.status}, Details: ${errorDetails}`);
    }

    console.log('Data successfully stored on OSF');
  } catch (error) {
    console.error('Error storing data on OSF:', error);
    throw error;  // Re-throw the error so it can be caught in endExperiment
  }
}

// End the experiment and upload data to OSF
function endExperiment() {
  console.log("endExperiment function is called...");
  const experimentData = jsPsych.data.get().values();
  console.log("Collected experiment data:", experimentData);

  // Log each trial's data for further confirmation
  experimentData.forEach((trial, index) => {
    console.log(`Trial ${index + 1} data:`, trial);
  });

  console.log("Attempting to store data on OSF...");
  storeDataOnOSF(experimentData, 'csv')
    .then(() => {
      console.log("Data storage process completed");
      // End the experiment after data is stored
      jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
    })
    .catch(error => {
      console.error("Error in storeDataOnOSF:", error);
      // End the experiment even if there's an error
      jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
    });
}

// Add end experiment screen
const thankYouTrial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p>Thank you for participating! Press the button to finish.</p>`,
  choices: ["Finish"],
  on_load: function() {
    console.log("Thank you trial loaded");
  },
  on_finish: function () {
    console.log("Thank you trial finished, calling endExperiment function now...");
    endExperiment();  // This will trigger the data storage process
    console.log("endExperiment function called, now ending the experiment");
    jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
  }
};

// Timer function for the entire experiment
function startExperimentTimer() {
  let timeRemaining = timeLimit;
  const timerElement = document.createElement('div');
  timerElement.id = 'timer';
  document.body.appendChild(timerElement);

  timerInterval = setInterval(() => {
    timeRemaining--;
    timerElement.textContent = `Time left: ${timeRemaining} s`;

    if (timeRemaining <= 0) {
      clearInterval(timerInterval);
      console.log("Time is up, ending experiment...");
      endExperiment();
      // Remove the jsPsych.endExperiment call from here
    }
  }, 1000);
}

// Start button screen
const startExperiment = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p style="font-size: 28px;">Let's begin!</p>`,
  choices: ["Start"],
  button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
  on_finish: () => {
    experimentStartTime = performance.now();  // Set the start time
    startExperimentTimer();  // Start the timer for the entire experiment
  }
};

// Run the experiment
async function runExperiment() {
  setGlobalStyles();

  const osfToken = await loadOSFToken();
  const bigramPairs = await loadBigramPairs();
  if (bigramPairs.length === 0) {
    jsPsych.end();  // End experiment if no bigram pairs available
    return;
  }

  const randomizedBigramPairs = jsPsych.randomization.shuffle(bigramPairs);
  const timeline = [];

  // Add start screen to timeline
  timeline.push(startExperiment);

  // Create trials for each pair of bigrams
    // Create trials for each pair of bigrams
    randomizedBigramPairs.forEach(([bigram1, bigram2], index) => {
      timeline.push(createTypingTrial(bigram1, [bigram1, bigram2], `trial-${index + 1}-1`));
      timeline.push(createTypingTrial(bigram2, [bigram1, bigram2], `trial-${index + 1}-2`));
      timeline.push(createComfortChoiceTrial(bigram1, bigram2, index + 1));  // Add the comfort choice trial
    });
  
    timeline.push(thankYouTrial);  // Add the thank-you screen to the end of the timeline
  
    // Run the timeline
    console.log("Running experiment timeline...");
    jsPsych.run(timeline);
}

// Start the experiment
runExperiment();