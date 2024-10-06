import { initJsPsych } from 'jspsych';
import htmlButtonResponse from '@jspsych/plugin-html-button-response';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import 'jspsych/css/jspsych.css';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables for countdown timer and required correct repetitions
let experimentConfig = {
  requiredCorrectRepetitions: 3,  // Default requirement to type the bigram correctly 3 times
  timeLimit: 10,  // Timer default time limit of 10 seconds for the entire experiment
  useTimer: false,  // Default to not using the timer
  practiceOnly: false,  // If true, only run the practice set
  randomizePairOrder: true,  // If true, randomize the order of bigram pairs
  randomizeBigramsWithinPairs: false,  // If true, randomize the sequence of bigrams within each pair
  trainingBigramFile: 'bigram_tables/bigram_3pairs_LH.csv',  // Default filename for training bigram pairs
  //trainingBigramFile: 'bigram_tables/bigram_3pairs_RH.csv',  // Default filename for training bigram pairs
  mainBigramFile: 'bigram_tables/bigram_2x80pairs_LH.csv'  // Default filename for main bigram pairs
  //mainBigramFile: 'bigram_tables/bigram_2x80pairs_RH.csv'  // Default filename for main bigram pairs
};

let experimentStartTime;
let timerInterval;

// Prolific completion URL with placeholder for the completion code
const PROLIFIC_COMPLETION_URL = "https://app.prolific.co/submissions/complete?cc=";
// Completion code for successful completion
const COMPLETION_CODE = "C1CL3V94";
// Completion code for no consent
const NO_CONSENT_CODE = "C15846F6";

// Get Prolific ID from URL parameters
let prolificID = '';
function getUrlParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}
prolificID = getUrlParam('PROLIFIC_PID') || 'unknown';

// Function to redirect to Prolific
function redirectToProlific(code) {
  console.log(`Redirecting to Prolific with code: ${code}`);
  window.location.href = PROLIFIC_COMPLETION_URL + code;
}

// Load OSF API token
async function loadOSFToken() {
  try {
    const response = await fetch('./token.json');
    const data = await response.json();
    console.log('OSF API token loaded');  // Log the loaded token
    return data.osf_api_token;
  } catch (error) {
    console.error('Error loading OSF token:', error);
    return null;
  }
}

// Load bigram pairs from a CSV file or text source
async function loadBigramPairs(trainingFile, mainFile) {
  async function loadFile(filename) {
    try {
      const response = await fetch(`./${filename}`);
      const csvText = await response.text();
      return csvText.split('\n').map(row => row.trim()).filter(row => row)
        .map(row => {
          // Use a regex to split the row, preserving commas within quotes
          return row.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g)
                    .map(entry => entry.replace(/^"(.*)"$/, '$1').trim());
        });
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      return [];
    }
  }

  const introductoryPairs = await loadFile(trainingFile);
  const mainPairs = await loadFile(mainFile);

  return { introductoryPairs, mainPairs };
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

// First informational page
const keyboardLayoutInfo = {
  type: htmlButtonResponse,
  stimulus: `
    <div class='instruction'> 
      <p>It is expected that your keyboard has the following character layout:</p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/qwerty-layout.jpg" width="500" style="max-width: 100%;">
      </div>
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// Second informational page
const typingInstructionsInfo = {
  type: htmlButtonResponse,
  stimulus: `
    <div class='instruction'> 
      <p>You will be asked to <strong>touch type</strong> pairs of letters.
         Touch type as you normally would, with left fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>A</span><span id=keystroke>S</span><span id=keystroke>D</span><span id=keystroke>F</span></span>
         and right fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>J</span><span id=keystroke>K</span><span id=keystroke>L</span><span id=keystroke>;</span></span>
         </p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/touchtype.jpg" width="500" style="max-width: 100%;">
      </div>
      <p>For instance, you will be asked to type 
        <span style="white-space: nowrap;">
          <span id=keystroke>a</span><span id=keystroke>b</span> 
          <span id=keystroke> </span>
          <span id=keystroke>c</span><span id=keystroke>d</span>
          <span id=keystroke> </span>
          <span id=keystroke>a</span><span id=keystroke>b</span>
          <span id=keystroke> </span>
          <span id=keystroke>c</span><span id=keystroke>d</span>
          <span id=keystroke> </span>          
          <span id=keystroke>a</span><span id=keystroke>b</span> 
          <span id=keystroke> </span>
          <span id=keystroke>c</span><span id=keystroke>d</span>
        </span>
      </p>
      <p>If you type this correctly, then you will be asked which letter pair 
        <span style="white-space: nowrap;"> &mdash; <span id=keystroke>a</span><span id=keystroke>b</span> or 
        <span id=keystroke>c</span><span id=keystroke>d</span> &mdash; </span> 
        is <b>easier</b> (more comfortable) for you to type. 
        Sometimes the two pairs may seem equally easy, 
        but please choose the one that is even slightly easier.</p> 
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// Consent trial
const consentTrial = {
  type: htmlButtonResponse,
  stimulus: `
    <div class='instruction' style='text-align: left; max-width: 800px; margin: 0 auto;'> 
      <h2 style='text-align: center;'>Welcome</h2>
      <dl>
          <dt>Purpose</dt>
          <dd>The purpose of this study is to determine how comfortable different pairs of keys 
          are to type on computer keyboards to inform the design of future keyboard layouts.</dd>

          <dt>Expectations</dt>
          <dd>It is expected that you will be <b>touch typing</b> on a <b>QWERTY desktop computer keyboard</b>.</dd>

          <dt>Procedures</dt>
          <dd>If you choose to participate, you will be repeatedly asked to type two new pairs 
          of letters and report which pair is easier (more comfortable) to type. </dd>

          <dt>Risks</dt>
          <dd>There are no anticipated risks or discomforts from this research that 
          you would not normally have when typing on your own keyboard.</dd>

          <dt>Benefits</dt>
          <dd>There are no anticipated benefits to you from this research.</dd>

          <dt>Compensation</dt>
          <dd>If you decide to participate, you will be compensated for your participation.</dd>

          <dt>Participation</dt>
          <dd>Taking part or not in this research study is your decision. 
          You can decide to participate and then change your mind at any point.</dd>

          <dt>Contact Information</dt>
          <dd>If you have any questions about the purpose, procedures, or any other issues 
          related to this research study you may contact the Principal Investigator, 
          Dr. Arno Klein, at arno.klein@childmind.org. </dd>
      </dl>
      <p style='text-align: center; font-weight: bold; margin-top: 20px;'>
        Do you consent to participate in this study? <br> You must be 18 years of age or older to participate.
      </p>
    </div>
  `,
  choices: ["I consent", "I do not consent"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>',
  on_finish: function(data) {
    if (data.response === 1) {  // "I do not consent" is selected
      // If consent is not given, redirect to Prolific with the no consent code
      redirectToProlific(NO_CONSENT_CODE);
    }
  }
};

// Function to update the color of individual letters as they're typed
function updateLetterColors(index, color, bold = false) {
  const letterSpans = document.querySelectorAll('.letter');
  if (letterSpans[index]) {
    letterSpans[index].style.color = color;
    letterSpans[index].style.fontWeight = bold ? 'bold' : 'normal';
  }
}

// Function to flash all letters red when a mistake is made
function flashAllLettersRed() {
  const letterSpans = document.querySelectorAll('.letter');
  letterSpans.forEach(span => {
    span.style.color = 'red';
    span.style.fontWeight = 'normal';  // Reset font weight
  });

  setTimeout(() => {
    letterSpans.forEach(span => {
      span.style.color = '';
      span.style.fontWeight = 'normal';
    });
  }, 500);  // Flash red for 500ms
}

function createTypingTrial(bigram1, bigram2, trialId, repetitions) {
  let keyData = [];
  let typedSequence = "";
  const trialStartTime = performance.now();
  let trialCompleted = false;
  const fullSequence = (bigram1 + ' ' + bigram2 + ' ').repeat(repetitions).trim();

  function handleKeyPress(event) {
    if (trialCompleted) return;

    let typedKey = event.key.toLowerCase();
    const expectedKey = fullSequence[typedSequence.length];
    const keydownTime = performance.now() - trialStartTime;

    if (event.key === 'Shift' || event.key.length > 1) {
      return;
    }

    if (typedKey === expectedKey) {
      typedSequence += typedKey;
      updateLetterColors(typedSequence.length - 1, 'green', true);  // Added true for bold
      keyData.push({
        expectedKey: expectedKey,
        typedKey: typedKey,
        keydownTime: keydownTime.toFixed(2),
        chosenBigram: "",
        unchosenBigram: ""
      });

      if (typedSequence === fullSequence) {
        trialCompleted = true;
        setTimeout(() => {
          jsPsych.finishTrial({
            keyData: keyData,
            task: 'typing',
            bigramPair: `${bigram1}, ${bigram2}`,
            fullSequence: fullSequence,
            trialId: trialId
          });
        }, 1000);
      }
    } else {
      flashAllLettersRed();
      typedSequence = "";
      keyData = [];
    }
  }

  return {
    type: htmlKeyboardResponse,
    stimulus: `
    <div class="jspsych-content-wrapper">
      <div class="jspsych-content">
        <p>Type the following letter pairs separated by a space:</p>
        <p id="sequence" style="font-size: 24px; letter-spacing: 2px;">
          ${fullSequence.split('').map(letter => `<span class="letter">${letter}</span>`).join('')}
        </p>
      </div>
    </div>`,
    choices: "NO_KEYS",
    trial_duration: null,
    on_load: function() {
      document.addEventListener('keydown', handleKeyPress);
    },
    on_finish: function(data) {
      document.removeEventListener('keydown', handleKeyPress);
      data.keyData = keyData;
      data.task = 'typing';
      data.bigramPair = `${bigram1}, ${bigram2}`;
      data.fullSequence = fullSequence;
      data.trialId = trialId;
      jsPsych.data.write(data);
      console.log("Typing Trial Data:", data);
    }
  };
}

// Comfort choice trial
function createComfortChoiceTrial(bigram1, bigram2, trialIndex) {
  return {
    type: htmlButtonResponse,
    stimulus: `<p style="font-size: 28px;">Which pair was easier to type?</p>`,
    choices: [bigram1, bigram2],
    button_html: '<button class="jspsych-btn comfort-choice-button">%choice%</button>',
    data: {
      task: 'comfort_choice',
      bigram1: bigram1,
      bigram2: bigram2,
      trialId: `trial-${trialIndex}-choice`
    },
    on_finish: function (data) {
      let chosenBigram = "";
      let unchosenBigram = "";
      if (data.response === 0) {
        chosenBigram = bigram1;
        unchosenBigram = bigram2;
      } else if (data.response === 1) {
        chosenBigram = bigram2;
        unchosenBigram = bigram1;
      }

      // Update the previous two typing trials with the chosen bigram
      const allData = jsPsych.data.get().values();
      const currentTrialIndex = allData.length - 1;
      
      for (let i = currentTrialIndex - 1; i >= currentTrialIndex - 2; i--) {
        if (allData[i].task === 'typing' && allData[i].keyData) {
          allData[i].keyData.forEach(keyEvent => {
            keyEvent.chosenBigram = chosenBigram;
            keyEvent.unchosenBigram = unchosenBigram;
          });
        }
      }
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
  const csvHeaders = ['trialId', 'bigramPair', 'bigramPairSequence', 'bigram', 'keyPosition', 'expectedKey', 'typedKey', 'keydownTime', 'chosenBigram', 'unchosenBigram'];
  let csvContent = csvHeaders.join(',') + '\n';

  let mainTrialCounter = 1;

  data.forEach(trial => {
    if (trial.task === 'typing' && trial.keyData) {
      const bigramPair = trial.bigramPair;
      const bigramPairSequence = trial.fullSequence;
      const bigrams = bigramPair.split(', ');

      let validKeyCounter = 0;

      // Determine if this is an intro trial or main trial
      let trialId;
      if (trial.trialId.startsWith('intro-trial-')) {
        trialId = trial.trialId;  // Keep the original intro trial ID
      } else {
        trialId = `trial${mainTrialCounter}`;
        mainTrialCounter++;
      }

      trial.keyData.forEach((keyEvent) => {
        // Skip rows with empty expectedKey or space
        if (!keyEvent.expectedKey || keyEvent.expectedKey === ' ') {
          return;
        }

        validKeyCounter++;
        const bigramIndex = Math.floor((validKeyCounter - 1) / 2) % 2;
        const currentBigram = bigrams[bigramIndex];
        const keyPosition = (validKeyCounter % 2 === 1) ? 1 : 2;

        const row = [
          escapeCSVField(trialId),
          escapeCSVField(bigramPair),
          escapeCSVField(bigramPairSequence),
          escapeCSVField(currentBigram),
          keyPosition,
          escapeCSVField(keyEvent.expectedKey),
          escapeCSVField(keyEvent.typedKey),
          keyEvent.keydownTime !== undefined ? keyEvent.keydownTime : '',
          escapeCSVField(keyEvent.chosenBigram || ''),
          escapeCSVField(keyEvent.unchosenBigram || '')
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
  const createFileUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=experiment_data_${prolificID}_${Date.now()}.${fileExtension}`;

  let fileData;
  let contentType;

  if (format === 'csv') {
    console.log("Converting data to CSV format...");
    fileData = convertToCSV(data);  // Use the new CSV conversion logic
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
    throw error;
  }
}

// End the experiment and upload data to OSF
function endExperiment() {
  const experimentData = jsPsych.data.get().values();
  console.log("All experiment data:", experimentData);

  // Filter out any empty or invalid data
  const validData = experimentData.filter(trial => trial.task === 'typing' && trial.keyData && trial.keyData.length > 0);
  console.log("Valid data for CSV:", validData);

  storeDataOnOSF(validData, 'csv')
    .then(() => {
      redirectToProlific(COMPLETION_CODE);
    })
    .catch(error => {
      console.error("Error in storeDataOnOSF:", error);
      redirectToProlific(COMPLETION_CODE);
    });
}

// Add end experiment screen
const thankYouTrial = {
  type: htmlButtonResponse,
  stimulus: `<p>Thank you for participating! <br>The experiment is now complete.</p>`,
  choices: ["Finish"],
  on_load: function() {
    console.log("Thank you trial loaded");
  },
  on_finish: function () {
    console.log("Thank you trial finished, calling endExperiment function now...");
    endExperiment();
  }
};

// Timer function for the entire experiment
function startExperimentTimer() {
  if (!experimentConfig.useTimer) {
    console.log("Timer is disabled.");
    return;  // Exit the function if the timer is not to be used
  }

  let timeRemaining = experimentConfig.timeLimit;
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
    }
  }, 1000);
}

// Start button screen
const startExperiment = {
  type: htmlButtonResponse,
  stimulus: function() {
    let stimulusText = `<p style="font-size: 28px;">Ready to start?</p>`;
    if (experimentConfig.useTimer) {
      stimulusText += `<p style="font-size: 24px;">You will have ${experimentConfig.timeLimit} seconds to complete the experiment.</p>`;
    }
    stimulusText += `<p style="font-size: 24px;">Press the button when you're ready to begin!</p>`;
    return stimulusText;
  },
  choices: ["Start"],
  button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
  on_finish: () => {
    experimentStartTime = performance.now();  // Set the start time
    if (experimentConfig.useTimer) {
      startExperimentTimer();  // Start the timer for the entire experiment (if enabled)
    }
  }
};

// Run the experiment
// Run the experiment
async function runExperiment(options = {}) {
  // Update experimentConfig with provided options
  Object.assign(experimentConfig, options);

  setGlobalStyles();

  const osfToken = await loadOSFToken();
  const { introductoryPairs, mainPairs } = await loadBigramPairs(
    experimentConfig.trainingBigramFile,
    experimentConfig.mainBigramFile
  );
  
  if (introductoryPairs.length === 0 || (mainPairs.length === 0 && !experimentConfig.practiceOnly)) {
    jsPsych.endExperiment('Error loading bigram pairs');
    return;
  }

  // Apply randomization based on configuration
  let processedMainPairs = mainPairs;
  if (experimentConfig.randomizePairOrder) {
    processedMainPairs = jsPsych.randomization.shuffle(processedMainPairs);
  }
  if (experimentConfig.randomizeBigramsWithinPairs) {
    processedMainPairs = processedMainPairs.map(pair => jsPsych.randomization.shuffle(pair));
  }

  const timeline = [];

  // Add consent trial to timeline
  timeline.push(consentTrial);

  // Create the experiment timeline
  const experimentTimeline = {
    timeline: [
      keyboardLayoutInfo,
      typingInstructionsInfo,
      startExperiment,
      // Introductory pairs (always in the same order)
      ...introductoryPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, bigram2, `intro-trial-${index + 1}`, experimentConfig.requiredCorrectRepetitions),
        createComfortChoiceTrial(bigram1, bigram2, `intro-${index + 1}`)
      ]),
    ],
    conditional_function: function() {
      // Only run this timeline if consent was given (i.e., the first option was selected)
      return jsPsych.data.get().last(1).values()[0].response === 0;
    }
  };

  // Add transition screen and main pairs if not practiceOnly
  if (!experimentConfig.practiceOnly) {
    experimentTimeline.timeline.push(
      {
        type: htmlButtonResponse,
        stimulus: `<p>Great job! You've completed the practice session.<br>
                   Now we'll move on to the main part of the experiment.</p>`,
        choices: ['Continue'],
      },
      ...processedMainPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, bigram2, `main-trial-${index + 1}`, experimentConfig.requiredCorrectRepetitions),
        createComfortChoiceTrial(bigram1, bigram2, `main-${index + 1}`)
      ])
    );
  }

  // Add thank you trial at the end
  experimentTimeline.timeline.push(thankYouTrial);

  timeline.push(experimentTimeline);

  // Run the timeline
  console.log("Running experiment timeline...");
  jsPsych.run(timeline);
}

// Start the experiment with options
runExperiment({
  practiceOnly: experimentConfig.practiceOnly,
  useTimer: experimentConfig.useTimer,
  timeLimit: experimentConfig.timeLimit,
  requiredCorrectRepetitions: experimentConfig.requiredCorrectRepetitions,
  randomizePairOrder: experimentConfig.randomizePairOrder,
  randomizeBigramsWithinPairs: experimentConfig.randomizeBigramsWithinPairs,
  trainingBigramFile: experimentConfig.trainingBigramFile,
  mainBigramFile: experimentConfig.mainBigramFile
});