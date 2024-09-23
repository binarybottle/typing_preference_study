import { initJsPsych } from 'jspsych';
import htmlButtonResponse from '@jspsych/plugin-html-button-response';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import 'jspsych/css/jspsych.css';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables and configuration
let experimentConfig = {
  requiredCorrectRepetitions: 3,
  timeLimit: 10,
  useTimer: false,
  practiceOnly: false,
  randomizePairOrder: true,
  randomizeBigramsWithinPairs: false,
  trainingBigramFile: 'bigram_tables/bigram_3pairs_LH.csv',
  mainBigramFile: 'bigram_tables/bigram_2x80pairs_LH.csv',
  character_list: 'abcdefghijklmnopqrstuvwxyz,./',
  ncharacters: 20,
};

let experimentStartTime;
let timerInterval;

// Prolific completion URL with placeholder for the completion code
const PROLIFIC_COMPLETION_URL = "https://app.prolific.co/submissions/complete?cc=";
const COMPLETION_CODE = "C1CL3V94";
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
    console.log('OSF API token loaded');
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
      redirectToProlific(NO_CONSENT_CODE);
    }
  }
};

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
      <p>You will be asked to <strong>touch type</strong> text containing pairs of letters.
         Touch type as you normally would, with left fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>A</span><span id=keystroke>S</span><span id=keystroke>D</span><span id=keystroke>F</span></span>
         and right fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>J</span><span id=keystroke>K</span><span id=keystroke>L</span><span id=keystroke>;</span></span>
         </p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/touchtype.jpg" width="500" style="max-width: 100%;">
      </div>
      <p>After typing the text, you will be asked which pair of letters was easier to type,
         and how much easier it was.</p>
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// New function to generate random text with interspersed bigrams
function generateTextWithBigrams(bigram1, bigram2, ncharacters) {
  let text = '';
  let bigrams = [
    { bigram: bigram1, color: '#0072B2', count: 0 },
    { bigram: bigram2, color: '#D55E00', count: 0 }
  ];
  let bigramPositions = [];

  // Generate base text
  for (let i = 0; i < ncharacters; i++) {
    if (i > 0 && i % (3 + Math.floor(Math.random() * 6)) === 0) {
      text += ' ';
    } else {
      text += experimentConfig.character_list[Math.floor(Math.random() * experimentConfig.character_list.length)];
    }
  }

  // Insert bigrams
  while (bigrams[0].count < 3 || bigrams[1].count < 3) {
    let bigramIndex = Math.floor(Math.random() * 2);
    if (bigrams[bigramIndex].count < 3) {
      let position;
      do {
        position = Math.floor(Math.random() * (text.length - 1));
      } while (bigramPositions.some(pos => Math.abs(pos - position) < 2));
      
      // Insert the full bigram
      text = text.slice(0, position) + bigrams[bigramIndex].bigram + text.slice(position);
      bigramPositions.push(position);
      bigrams[bigramIndex].count++;
    }
  }

  return { text, bigrams };
}

// Modified typing trial
function createTypingTrial(bigram1, bigram2, trialId) {
  const { text, bigrams } = generateTextWithBigrams(bigram1, bigram2, experimentConfig.ncharacters);
  let typedSequence = "";
  let keyData = [];
  const trialStartTime = performance.now();

  function handleKeyPress(event) {
    let typedKey = event.key.toLowerCase();
    const expectedKey = text[typedSequence.length];
    const keydownTime = performance.now() - trialStartTime;

    if (event.key === 'Shift' || event.key.length > 1) {
      return;
    }

    if (typedKey === expectedKey) {
      typedSequence += typedKey;
      updateLetterColors(typedSequence.length - 1, 'gray');
      keyData.push({
        expectedKey: expectedKey,
        typedKey: typedKey,
        isCorrect: true,
        keydownTime: keydownTime.toFixed(2)
      });

      if (typedSequence === text) {
        setTimeout(() => {
          jsPsych.finishTrial({
            keyData: keyData,
            task: 'typing',
            bigramPair: `${bigram1}, ${bigram2}`,
            fullText: text,
            trialId: trialId
          });
        }, 500);
      }
    } else {
      keyData.push({
        expectedKey: expectedKey,
        typedKey: typedKey,
        isCorrect: false,
        keydownTime: keydownTime.toFixed(2)
      });
    }
  }

  return {
    type: htmlKeyboardResponse,
    stimulus: function() {
      let styledText = '';
      for (let i = 0; i < text.length; i++) {
        let style = '';
        for (let bigram of bigrams) {
          if (text.substr(i, 2) === bigram.bigram) {
            style = `color: ${bigram.color}; font-weight: bold;`;
            break;
          }
        }
        styledText += `<span style="${style}">${text[i]}</span>`;
      }

      return `
        <div class="jspsych-content-wrapper">
          <div class="jspsych-content">
            <p>Type the following text:</p>
            <p id="text-to-type" style="font-size: 24px; letter-spacing: 2px;">
              ${styledText}
            </p>
          </div>
        </div>`;
    },
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
      data.fullText = text;
      data.trialId = trialId;
    }
  };
}

// New comfort choice trial with buttons and slider
function createComfortChoiceTrial(bigram1, bigram2, trialId) {
  const buttonTrial = {
    type: htmlButtonResponse,
    stimulus: `<p style="font-size: 28px;">Which letter pair was easier to type?</p>`,
    choices: [bigram1, bigram2],
    button_html: button => `<button class="jspsych-btn comfort-choice-button" style="color: ${button === bigram1 ? '#0072B2' : '#D55E00'}">%choice%</button>`,
    data: {
      task: 'comfort_choice_button',
      bigram1: bigram1,
      bigram2: bigram2,
      trialId: trialId
    },
    on_finish: function(data) {
      data.chosenBigram = data.response === 0 ? bigram1 : bigram2;
      data.unchosenBigram = data.response === 0 ? bigram2 : bigram1;
    }
  };

  const customSliderTrial = {
    type: htmlButtonResponse,
    stimulus: function() {
      const { chosenBigram, unchosenBigram } = jsPsych.data.get().last(1).values()[0];
      return `
        <p style="font-size: 24px;">How much easier was <span style="color: ${chosenBigram === bigram1 ? '#0072B2' : '#D55E00'}">${chosenBigram}</span> to type than <span style="color: ${unchosenBigram === bigram1 ? '#0072B2' : '#D55E00'}">${unchosenBigram}</span>?</p>
        <div id="custom-slider" style="width: 500px; margin: 20px auto;">
          <input type="range" min="0" max="100" value="50" style="width: 100%;">
          <div style="display: flex; justify-content: space-between;">
            <span>slightly easier</span>
            <span>much easier</span>
          </div>
        </div>
        <p id="slider-value"></p>
      `;
    },
    choices: ["Submit"],
    on_load: function() {
      const slider = document.querySelector('#custom-slider input');
      const output = document.querySelector('#slider-value');
      slider.oninput = function() {
        output.textContent = `Selected value: ${this.value}`;
      };
    },
    on_finish: function(data) {
      const sliderValue = document.querySelector('#custom-slider input').value;
      data.sliderResponse = parseInt(sliderValue) / 100; // Convert to 0-1 range
      data.task = 'comfort_choice_slider';
    }
  };

  return {
    timeline: [buttonTrial, customSliderTrial],
    data: {
      bigramPair: `${bigram1}, ${bigram2}`,
      trialId: trialId
    }
  };
}

// Function to update the color of individual letters as they're typed
function updateLetterColors(index, color) {
  const letterSpans = document.querySelectorAll('#text-to-type span');
  if (letterSpans[index]) {
    letterSpans[index].style.color = color;
  }
}

// Modified function to convert data to CSV format
function convertToCSV(data, fileType) {
  let csvHeaders, csvContent;

  if (fileType === 'raw') {
    csvHeaders = ['user_id', 'trialId', 'expectedKey', 'typedKey', 'isCorrect', 'keydownTime'];
    csvContent = csvHeaders.join(',') + '\n';

    data.forEach(trial => {
      if (trial.task === 'typing' && trial.keyData) {
        trial.keyData.forEach(keyEvent => {
          const row = [
            prolificID,
            trial.trialId,
            escapeCSVField(keyEvent.expectedKey),
            escapeCSVField(keyEvent.typedKey),
            keyEvent.isCorrect,
            keyEvent.keydownTime
          ];
          csvContent += row.join(',') + '\n';
        });
      }
    });
  } else if (fileType === 'summary') {
    csvHeaders = ['user_id', 'trialId', 'textString', 'chosenBigram', 'unchosenBigram', 'chosenBigramTime', 'unchosenBigramTime', 'comfortRating'];
    csvContent = csvHeaders.join(',') + '\n';

    data.forEach((trial, index, array) => {
      if (trial.task === 'comfort_choice_slider') {
        const typingTrial = array[index - 2];
        const choiceTrial = array[index - 1];

        if (typingTrial && choiceTrial) {
          const chosenBigram = choiceTrial.chosenBigram;
          const unchosenBigram = choiceTrial.unchosenBigram;
          const bigramTimes = calculateBigramTimes(typingTrial.keyData, chosenBigram, unchosenBigram);

          const row = [
            prolificID,
            trial.trialId,
            escapeCSVField(typingTrial.fullText),
            escapeCSVField(chosenBigram),
            escapeCSVField(unchosenBigram),
            bigramTimes.chosenBigramTime,
            bigramTimes.unchosenBigramTime,
            trial.sliderResponse
          ];
          csvContent += row.join(',') + '\n';
        }
      }
    });
  }

  return csvContent;
}

// Helper function to escape commas and wrap fields in quotes if necessary
function escapeCSVField(field) {
  if (typeof field === 'string' && field.includes(',')) {
    return `"${field.replace(/"/g, '""')}"`;  // Escape double quotes by doubling them
  }
  return field;
}

// Helper function to calculate median bigram times
function calculateBigramTimes(keyData, chosenBigram, unchosenBigram) {
  function getMedianTime(bigram) {
    const times = [];
    for (let i = 0; i < keyData.length - 1; i++) {
      if (keyData[i].expectedKey + keyData[i + 1].expectedKey === bigram) {
        times.push(keyData[i + 1].keydownTime - keyData[i].keydownTime);
      }
    }
    times.sort((a, b) => a - b);
    return times.length > 0 ? times[Math.floor(times.length / 2)] : null;
  }

  return {
    chosenBigramTime: getMedianTime(chosenBigram),
    unchosenBigramTime: getMedianTime(unchosenBigram)
  };
}

// Modified function to store data on OSF
async function storeDataOnOSF(data) {
  const osfToken = await loadOSFToken();
  if (!osfToken) {
    console.error('Error: OSF API token not available. Data will not be stored on OSF.');
    return;
  }

  const osfNodeId = "jf8sc";
  const rawDataUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=raw_data_${prolificID}_${Date.now()}.csv`;
  const summaryDataUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=summary_data_${prolificID}_${Date.now()}.csv`;

  const rawCsvData = convertToCSV(data, 'raw');
  const summaryCsvData = convertToCSV(data, 'summary');

  try {
    await uploadToOSF(rawDataUrl, rawCsvData, osfToken);
    await uploadToOSF(summaryDataUrl, summaryCsvData, osfToken);
    console.log('Data successfully stored on OSF');
  } catch (error) {
    console.error('Error storing data on OSF:', error);
    throw error;
  }
}

async function uploadToOSF(url, data, token) {
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'text/csv',
    },
    body: data,
  });

  if (!response.ok) {
    const errorDetails = await response.text();
    throw new Error(`Upload error! Status: ${response.status}, Details: ${errorDetails}`);
  }
}

// Timer function for the entire experiment
function startExperimentTimer() {
  if (!experimentConfig.useTimer) {
    console.log("Timer is disabled.");
    return;
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
    experimentStartTime = performance.now();
    if (experimentConfig.useTimer) {
      startExperimentTimer();
    }
  }
};

// End the experiment and upload data to OSF
function endExperiment() {
  const experimentData = jsPsych.data.get().values();
  console.log("All experiment data:", experimentData);

  // Filter out any empty or invalid data
  const validData = experimentData.filter(trial => 
    (trial.task === 'typing' && trial.keyData && trial.keyData.length > 0) ||
    trial.task === 'comfort_choice_button' ||
    trial.task === 'comfort_choice_slider'
  );
  console.log("Valid data for CSV:", validData);

  storeDataOnOSF(validData)
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

// Modified runExperiment function
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
        createTypingTrial(bigram1, bigram2, `intro-trial-${index + 1}`),
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
        createTypingTrial(bigram1, bigram2, `main-trial-${index + 1}`),
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
  mainBigramFile: experimentConfig.mainBigramFile,
  character_list: experimentConfig.character_list,
  ncharacters: experimentConfig.ncharacters
});