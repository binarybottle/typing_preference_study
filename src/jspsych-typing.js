import { twoLetterPair, checkEmpty, createTable, calQuantile, makeMultipliers } from "./utils";
import HtmlKeyboardResponsePlugin from "@jspsych/plugin-html-keyboard-response";
import PreloadPlugin from "@jspsych/plugin-preload";
import SurveyMultiSelectPlugin from "@jspsych/plugin-survey-multi-select";
import FullscreenPlugin from "@jspsych/plugin-fullscreen";
import InstructionsPlugin from "@jspsych/plugin-instructions";
import jspsychKeyboardDisplay from "./jspsych-keyboard-display";
import SurveyMultiChoicePlugin from "@jspsych/plugin-survey-multi-choice";
import HtmlButtonResponsePlugin from "@jspsych/plugin-html-button-response";
import SurveyTextPlugin from "@jspsych/plugin-survey-text";

export const DICT = {
    'preload': PreloadPlugin,
    'checkbox': SurveyMultiSelectPlugin,
    'choice': SurveyMultiChoicePlugin,
    'fullscreen': FullscreenPlugin,
    'instruct': InstructionsPlugin,
    'response': HtmlKeyboardResponsePlugin,
    'display': jspsychKeyboardDisplay,
    'button': HtmlButtonResponsePlugin,
    'text': SurveyTextPlugin,
}
/** 
* @summary The function does the following:
    a. After a response, update the score and the history of keyboard presses.
    b. Check if the keyboard press is valid.
    c. Record time lapses before making the first response.
    d. Assign ${info.rt} to ${response.rt_valid}.
    e. Compare the typed key with the expected key from ${trial.choices[i]} (i for alternating purpose) to update the score.
*
* @param {ParamDataTypeHere} parameterNameHere - Brief description of the parameter here. Note: For other notations of data types, please refer to JSDocs: DataTypes command.
* @param {object} info          - response time and keyboard behavior of participants responding to this stimulus
* @param {object} response      - statistics to record and update at each round, e.g., rt_valid for response time of valid pressing
* @param {object} trail         - timeline object containing all nececessary components to run this trial
* @param {Array} response_history       - an array of history keyboard pressings
* @param {object} counter       - a counter used to determine the expected keyboard behaviors
* @param {html-element} display_html    - html element, i.e., the content of displayed stimulus
* @param {function} end_trial   - a function to call before the end of this trial
*/

const multiplierArray1 = makeMultipliers();
const multiplierArray2 = makeMultipliers();
const multiplierArray = multiplierArray1.concat(multiplierArray2);

console.log(multiplierArray);

function keypressCallback(info, response, trial, response_history, counter, display_html, end_trial) {

    display_html.querySelector("#jspsych-html-keyboard-response-stimulus").className += " responded";
    if (response.score === 0) {
        response.rt_valid = info.rt;
    }
    if (response.typed === 0) {
        response.rt_typed = info.rt;
    }
    let i = counter.i % trial.choices.length;
    if (jsPsych.pluginAPI.compareKeys(info.key, trial.choices[i])) {
        // if participants press the correct key in correct order
        response.score++;
        counter.i++;
        // update html to show improved score
        $("#keypress-count").html(response.score.toString());
        // add a node to help display this choice
        response_history.push({key: info.key, color: 'green'})
    } else {
        // if participants press the wrong key or in wrong order
        // add a node to help display this choice
        response_history.push({key: info.key, color: 'red'})
    }
    /**
     * if trial.num_keypress_display > 0, display histories of keyboard pressings
     */
    if (trial.num_keypress_display > 0) {
        if (response_history.length > trial.num_keypress_display) {
            response_history.shift(); //remove first (oldestl) element from array
        }
        let keysList = document.getElementById("keypress-hist");
        keysList.innerHTML = '';
        response_history.forEach(function(hist){
            let li = document.createElement('li');
            li.textContent = hist.key;
            li.style.color = hist.color;
            keysList.appendChild(li);
        });
    }
    response.typed++;
    /**
     * end the trial if 
     */
    if (trial.response_ends_trial) { 
        end_trial()
    }
};

/**
 * This function helps to render jspsych compatible plugins from yaml configurations
 * 
 * @param
 */
export function renderPlugin({
    args = {}, on_start = undefined, on_load = undefined, on_finish = undefined,
    conditional_function = undefined, loop_function = undefined
} = {}) {
    args.type = DICT[args.type]
    args = {
        on_start: on_start, on_finish: on_finish, on_load: on_load, ...args
    }
    if ([conditional_function, loop_function].some(x => typeof x !== "undefined")) {
        return {
            timeline: args, conditional_function: conditional_function, loop_function: loop_function
        }
    }
    return args
}


export function bonusInstruction({
    instruction = undefined, example = undefined, questions = [], condition = "",
    randomize_order = true, data = {}, feedback_success = "success", feedback_failure = "failure",
} = {}) {

    let correct = false;
    const pages = [];
    const page_1 = instruction.title.replaceAll('${instruction_by_group}', instruction[condition]);
    const page_2 = instruction.info.replaceAll('${instruction_by_group}', instruction[condition]);
    pages.push(page_1);
    pages.push(page_2);
    const page_success = `<div><p class=instruction-title>If you score <b><span id='current-number'>212</span></b> and the target score is <b><span id='target-number'>200</span></b>, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <div class="bonus-2">
                <div class="bonus-3">Your score:</div>
                <div class="bonus-4"><span id="current-number">212</span></div>
            </div>
            <div class="bonus-2">
                <div class="bonus-3">Target score:</div>
                <div class="bonus-4" style="color: red"><span id="target-number">200</span></div>
            </div>
        </div>`;

    pages.push(page_success);

    const page_success_binary = `<div><p class=instruction-title>Then you'll see that you won a 5 cent bonus:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <div class="feedback-container">
                <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
                <div class="feedback-text"><p>You reached the target score!</p><span style="font-size: 75px; line-height:90px">+5</span></div>
            </div>
        </div>`;

    const page_success_cStrk = `<div><p class=instruction-title>Then you'll see the length of your current streak.<br>For example, after three consecutive completions, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <p>You reached the target score!</p>Current Streak: 3
        </div>`;

    const page_success_bStrk = [`<div><p class=instruction-title>Then you'll see the length of your current streak.<br>For example, after one completion, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <p>You reached the target score!</p>Current Streak: 1/3
        </div>`,

        `<div><p class=instruction-title>After two completions, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <p>You reached the target score!</p>Current Streak: 2/3
        </div>`,

        `<div><p class=instruction-title>After three completions, you'll see that you won a 15 cent bonus:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <div class="feedback-container">
                <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
                <div class="feedback-text"><p>You got a streak of 3!</p><span style="font-size: 75px; line-height:90px">+15</span></div>
            </div>
        </div>`];

    if (condition == "binary streak") {
        pages.push(...page_success_bStrk)
    } else if (condition == "continuous streak") {
        pages.push(page_success_cStrk) 
    } else if (condition == "binary") {
        pages.push(page_success_binary)
    };

    const page_fail = `<div><p class=instruction-title>If you score <b><span id='current-number'>188</span></b> and the target score is <b><span id='target-number'>200</span></b>, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <div class="bonus-2">
                <div class="bonus-3">Your score:</div>
                <div class="bonus-4"><span id="current-number">188</span></div>
            </div>
            <div class="bonus-2">
                <div class="bonus-3">Target score:</div>
                <div class="bonus-4" style="color: red"><span id="target-number">200</span></div>
            </div>
        </div>`;

    pages.push(page_fail);

    const page_fail_binary = `<div><p class=instruction-title>Then you'll see that you failed to win a bonus:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
        <div style="position: relative; text-align: center; width: 800px; height: 150px">
        <div class="feedback-text">
        <p>You missed the target score.</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>
        </div></div></div>`;

    const page_fail_bStrk = `<div><p class=instruction-title>Then you'll see that you failed to win a bonus:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
        <div style="position: relative; text-align: center; width: 800px; height: 150px">
        <div class="feedback-text">
        <p>Your streak was less than 3.</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>
        </div></div></div>`;

    const page_fail_cStrk = [`<div><p class=instruction-title>Then you'll see how much money you earned from your steak.<br>
        For example, if you miss the target score after achieving a streak of three, you'll see:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
            <div class="feedback-container">
                <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
                <div class="feedback-text"><p>Your streak was 3</p><span style="font-size: 75px; line-height:90px">+15</span></div>
            </div>
        </div>`,

        `<div><p class=instruction-title>However, if you miss the target score after failing to start a streak,<br>
        you'll see that you failed to win a bonus:</p></div>
        <div class="bonus-1" style="margin: 40px 0px">
        <div style="position: relative; text-align: center; width: 800px; height: 150px">
        <div class="feedback-text">
        <p>Your streak was 0</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>
        </div></div></div>`];

    if (condition == "binary streak") {
        pages.push(page_fail_bStrk)
    } else if (condition == "continuous streak") {
        pages.push(...page_fail_cStrk) 
    } else if (condition == "binary") {
        pages.push(page_fail_binary)
    };

    /*
    example.title.forEach( (title, index) => {
        title = example.header[index] + title;
        let display_html = $('<div />', {html: title});
        display_html.find('div#bonus-number').html(example[condition][Math.floor(index/2)]);
        display_html.find('span#target-number').html('200');
        display_html.find('span#current-number').html(index===0? '212' : '188');
        pages.push(display_html.html());
    })
    */
    
    const welcome = {
        type: InstructionsPlugin,
        pages: pages,
        show_clickable_nav: true
    };

    const comprehension = {
        type: SurveyMultiChoicePlugin,
        questions: questions.map(x => ({ ...x, required: true })),
        randomize_question_order: randomize_order,
        on_finish: function (data) {
            let { Q0, Q1, Q2 } = data.response;
            data.pass = correct = [
                Q0.includes("At least"),
                Q1.startsWith(condition === "binary streak" ? '0' : '10'),
                Q2.startsWith('15')
            ].every(Boolean)
        },
    };

    const feedback = {
        type: HtmlButtonResponsePlugin,
        stimulus: "",
        on_start: (trial) => {
            trial.stimulus = correct ? feedback_success : feedback_failure
        },
        choices: ["Continue"]
    };

    return {
        timeline: [welcome, comprehension, feedback],
        loop_function: (data) => {
            return !data.select('pass').values[0]; //or simple correct!
        },
        data: data
    }
}


/**
 * This class defines contents of practice phase
 * 
 * @param {numeric} numOfTrial           The number of total trials during the practice stage
 * @param {numeric} trial_duration       An integer indicates the time (in seconds) before time out
 * @param {numeric} num_keypress_display The number of past keyboard pressings to be displayed, hide if set to zero
 * @param {string[]} list                A list of two-letter pairs be serve as correct responses (in sequential order) at each practice trial
 * @param {object} data                  What additional data should the trial store
 * @param {numeric} fontsize             Fontsize of displayed stimulus
 */
export class practicePhase {
    constructor({ numOfTrial = 1, trial_duration = 5, fixation_duration = 3, num_keypress_display = 0, fontsize = "", no_prompt = false, show_stat = false, list = [], data = {}, true_random = undefined, prob_win = undefined } = {}) {
        this.numOfTrial = numOfTrial;
        this.num_keypress_display = num_keypress_display;
        this.data = data;
        this.fontsize = fontsize;
        this.no_prompt = no_prompt;
        this.show_stat = show_stat;
        this.true_random = true_random;
        this.prob_win = prob_win;
        if (list.length < numOfTrial) {
            // randomly generate ${numOfTrial} two-letter pairs
            const n = list.length;
            list = list.concat(twoLetterPair(numOfTrial-n));
        };
        this.list = list.slice(0, numOfTrial);
        this.trial_duration = trial_duration * 1000;
        this.fixation_duration = fixation_duration * 1000;
        this.phase = 'practice';
        this.trial_i = 0;
    }

    getFixation(timeline) {
        timeline.push({
            on_start: (trial) => {
                let [a, b] = trial.stimulus;
                trial.stimulus = `<p>type ${a} and then ${b} as many times as possible</p>`;
            },
            type: HtmlKeyboardResponsePlugin,
            stimulus: jsPsych.timelineVariable("choices"),
            choices: "NO_KEYS",
            trial_duration: this.fixation_duration,
            data: { phase: `${this.phase}_fixation` },
        });
        /*
        timeline.push({
            type: HtmlKeyboardResponsePlugin,
            stimulus: "<div style='font-size: 72px'>+</div>",
            choices: "NO_KEYS",
            trial_duration: 1000,
            data: { phase: `${this.phase}_fixation` },
        });
        */
    }

    getFeedback(timeline) {
        const feedback = [];
        const on_load = () => {
            const data = [...Object.values(jsPsych.data.get().filter({ phase: 'practice' }))][0];
            if (!checkEmpty(data)) {
                const selected_column = ['rt_valid', 'rt_typed', 'typed', 'score'];
                const table = createTable(data, selected_column);
                if (this.show_stat) {
                    $("div.jspsych-content").prepend(table);
                } else {
                    jsPsych.finishTrial();
                };
            };
        };
        feedback.push({
            type: HtmlButtonResponsePlugin,
            on_load: on_load,
            stimulus: "",
            data: { phase: "statistics" },
            choices: ["Continue"],
        });
        timeline.push({
            timeline: feedback,
            conditional_function: () => this.trial_i === this.numOfTrial
        })
    }

    getStimulusCallback(){
        const cb = [];
        cb.push({
            callback_function: keypressCallback, 
            accept_allkeys: true, 
        });
        return cb
    }

    getStimulusOnStart(trial){
        let [a, b] = trial.choices;
        trial.prompt = this.no_prompt ? "" : `<p id="display-prompt">type ${a} and then ${b} as many times as possible</p>`;
        this.trial_i++;
    }

    getTimelineVariable() {
        const result = [];
        this.list.forEach((pair) => {
            let [first, second] = pair.split(',');
            result.push({ choices: [first, second] })
        })
        return result
    }

    getStimulus(timeline) {
        timeline.push({
            on_start: (trial) => this.getStimulusOnStart(trial),
            type: jspsychKeyboardDisplay,
            stimulus: "",
            choices: jsPsych.timelineVariable("choices"),
            true_random: this.true_random,
            prob_win: this.prob_win,
            remain_time_display: true,
            num_keypress_display: this.num_keypress_display,
            response_ends_trial: false,
            data: this.data,
            trial_duration: () => this.trial_duration,
            response_callbacks: () => this.getStimulusCallback(),
        })
    }

    on_timeline_start() {
        $('div#jspsych-content').css({ "font-size": this.fontsize });
    }
    on_timeline_finish() {
        $('div#jspsych-content').css({ "font-size": "" });
    }

    getTrial() {
        const timeline = [];
        // push fixation
        this.getFixation(timeline);
        // push practice
        this.getStimulus(timeline);
        // push feedback
        this.getFeedback(timeline);
        return {
            timeline: timeline,
            timeline_variables: this.getTimelineVariable(),
            randomize_order: true,
            on_timeline_start: () => this.on_timeline_start(),
            on_timeline_finish: () => this.on_timeline_finish(),
            conditional_function: () => this.numOfTrial > 0,
        }
    }
}


function getDist(args){
    let {choice, quantile} = args;
    if (choice === "normal"){
        const {std} = args[choice];
        return (scores) => {
            const center = calQuantile(scores, quantile);
            return () => Math.floor(jsPsych.randomization.sampleNormal(center, std)) || center;
        }
    } else if (choice === "uniform"){
        const {range} = args[choice];
        return (scores) => {
            const center = calQuantile(scores, quantile);
            return () => jsPsych.randomization.randomInt(center-range, center+range) || center;
        }
    }
}

export class bonusPhase extends practicePhase {
    constructor({ numOfTrial = 1, fontsize = "", no_prompt = false, list = [], data = {}, target_dist = {}, target_min = 1, condition = undefined, feedback = undefined, early_stop = undefined, time = undefined, true_random = undefined, prob_win = undefined, first_trial_num = undefined} = {}) {
        const {trial_time, fix_time, score_time, reward_time} = time;
        super({ numOfTrial: numOfTrial, trial_duration: trial_time, fixation_duration: fix_time, fontsize: fontsize, no_prompt: no_prompt, list: list, data: data });
        this.target_dist = getDist(target_dist);
        this.target_min = target_min;
        this.practice_score = 0;
        this.phase = "bonus";
        this.condition = condition;
        this.early_stop = early_stop;
        this.score_time = score_time * 1000;
        this.reward_time = reward_time * 1000;
        this.feedback = feedback;
        this.true_random = true_random;
        this.prob_win = prob_win;
        this.multiplier = 1;
        this.delta = 20;
        this.trial_i = first_trial_num;
        this.reward_agent = (condition === "binary") ?
            new Binary() : (condition === "continuous streak") ?
            new ContinuousStreak() : new BinaryStreak()
    }

    on_timeline_start() {
        super.on_timeline_start()
        const practice_scores = jsPsych.data.get().filter({ phase: 'practice' }).select('score').values;
        this.target_dist = this.target_dist(practice_scores);
    }

    getStimulusCallback(){
        /**
         * other than every statistsics we recorded during the practice phase,
         * we also need to tell if the number of valid presses have surpassed target threshold
         */
        const cb = [];
        cb.push({
            callback_function: (info, response, trial, response_history, counter, display_html, end_trial) => {
                keypressCallback(info, response, trial, response_history, counter, display_html, end_trial);

                // draw delta on first click
                if (response.typed == 1) { this.delta = 1 + Math.floor(Math.random() * 20 ) };

                // make multiplier = 1 when score < delta and randomly draw new multiplier when score surpasses delta
                if (response.score <= this.delta) { 
                    this.multiplier = 1;
                } else if (response.score == this.delta + 1) {                    
                    this.multiplier = multiplierArray[this.trial_i - 1];
                };

                // compute new target score and, if true_random = true, make it the real target score 
                let targetScore = response.score + (this.multiplier * this.delta);
                if (this.true_random) { trial.data.target = targetScore };
                if (this.trial_i == this.numOfTrial) { trial.data.target = response.score + this.delta }; // lose on last trial

                console.log(`Trial Number = ${this.trial_i}`);

                if (response.score >= trial.data.target) {
                    console.log("Success!")
                    trial.data.success = true;
                    this.early_stop && end_trial(trial.data);
                }
            }, 
            accept_allkeys: true, 
        });
        return cb
    }

    getStimulusOnStart(trial) {
        /**
         * generate a target threshold number for this trial/round
         */
        super.getStimulusOnStart(trial);

        trial.data = {
            success: false,             //whether or not participants win this round
            target: Math.max(this.target_dist(), this.target_min),    //a default infinity is set by default
            ...trial.data
        }
    }

    getFeedback(timeline) {
        const {main_feedback, success_feedback, failure_feedback} = this.feedback;
        let success = 0, target = 0, current = 0, bonus = 0, streak = 0, bonus_msg = undefined;
        const make_page = function(duration, phase, on_start) {
            return {
                type: HtmlKeyboardResponsePlugin,
                stimulus: "",
                choices: "NO_KEYS",
                trial_duration: duration,
                data: { phase: phase }, 
                on_start: on_start
            }
        };
        const on_start = (trial) => {
            let stimulus = "";
            if (trial.data.phase === 'bonus_feedback_reward'){
                stimulus = success? success_feedback : failure_feedback;
            } else {
                stimulus = main_feedback;
                trial.data.bonus = bonus;
            }
            let display_html = $('<div />', {html: stimulus});
            display_html.find('div#feedback').html(bonus_msg);
            display_html.find('span#target-number').html(target);
            display_html.find('span#current-number').html(current);
            if (this.condition !== "binary" && !success) {
                const div = display_html.find('div#bonus-2');
                const streak_span = this.condition === 'binary streak'? `${streak}/3` : `${streak}`;
                streak === 0 
                    ? div.remove()
                    : div.html("Your streak was " + streak_span);
            }
            trial.stimulus = display_html.html();
            //trial.stimulus = `Congrats!`;
        };
        timeline.push({
            timeline: [
                make_page(this.score_time, 'bonus_feedback_score', on_start),
                make_page(this.reward_time, 'bonus_feedback_reward', on_start)
            ],
            on_timeline_start: () => {
                const response = jsPsych.data.getLastTrialData().trials[0];
                ({success, target, score: current} = response);
                bonus_msg = this.reward_agent.step(success);
                ({bonus, streak} = this.reward_agent.property);
                if (this.condition === "continuous streak" && this.trial_i === this.numOfTrial) {
                    // the bonus in the last round isn't counted when under this condition
                    bonus = +(5 * this.reward_agent.streak);
                }
            }
        });
    }
}

class Binary {

    constructor() {
        this.overall_bonus = 0;
        this.bonus = 0;
        this.streak_sofar = 0;
    }
    step(success) {
        this.bonus = this.score(success);
        this.overall_bonus += this.bonus;
        if (success) {
            return `<div class="feedback-container">
            <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
            <div class="feedback-text"><p>You reached the target score!</p><span style="font-size: 75px; line-height:90px">+${this.bonus}</span></div>
            </div>`

        } else {
            return `<p>You missed the target score.</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>`
        }
         
        //return `Bonus: + $${this.bonus.toFixed(2)}`;
    }
    score(success){
        return success? 5 : 0
    }
    get property(){
        return {
            streak: this.streak_sofar,
            bonus: this.bonus
        }
    }
}

class ContinuousStreak extends Binary {

    constructor() {
        super();
        this.streak = this.streak_sofar;
    }
    step(success) {
        this.streak_sofar = this.streak;
        this.bonus = this.score(success);
        // show streak length when increase or initiate a streak
        if(success){
            this.streak += 1;
            return `<p>You reached the target score!</p>Current Streak: ${this.streak}`;        
        // show bonus information only when break a streak, show streak length when cannot initiate
        } else {
            this.overall_bonus += this.bonus;
            if (this.streak_sofar === 0){
                // fail to initiate, after a failure
                return `<p>Your streak was 0</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>`;
            } else {
                // break a streak
                this.streak = 0;
                return `<div class="feedback-container">
                <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
                <div class="feedback-text"><p>Your streak was ${this.streak_sofar}</p><span style="font-size: 75px; line-height:90px">+${this.bonus}</span></div>
                </div>`
            }
        };
    }
    score(succcess){
        return succcess? 0 : this.streak_sofar*5
    }
}

class BinaryStreak extends ContinuousStreak {

    step(success){
        this.streak_sofar = this.streak;
        this.bonus = this.score(success);
        if(success){
            this.streak += 1;
            if (this.bonus) {
                this.overall_bonus += this.bonus;
                this.streak = 0;
                // streak complete: participants earn bonus money
                return `<div class="feedback-container">
                <img src="https://raw.githubusercontent.com/dennislx/jspsych-typing/main/public/img/coins.jpg">
                <div class="feedback-text"><p>You got a streak of 3!</p><span style="font-size: 75px; line-height:90px">+${this.bonus}</span></div>
                </div>`;            
            } else {
                // streak increase: show streak length
                return `<p>You reached the target score!</p>Current Streak: ${this.streak}/3`;  
            }
        } else {
            if (this.streak_sofar === 0){
                // fail to initiate (could be after a failure or after a complete streak)
                return `<p>Your streak was less than 3.</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>`;
            } else {
                this.streak = 0; 
                // fail to break
                //return `Bonus: + $${this.bonus.toFixed(2)}`
                return `<p>Your streak was less than 3.</p><span style="font-size: 75px; line-height:90px; font-weight: bold">+0</span>`;
            }
        }
    }
    score(success){
        return !success? 0 : ((this.streak_sofar+1)===3) ? 15 : 0
    }
}