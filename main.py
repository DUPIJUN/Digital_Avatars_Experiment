#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on April 20, 2026, at 00:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.3'
expName = 'main'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'gender': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (3072, 1920)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\32917\\OneDrive\\Desktop\\experiment_repos\\Experiment\\main.py',
        savePickle=False, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='cover',
            blendMode='avg', useFBO=True,
            units='pix',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'cover'
        win.units = 'pix'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instr" ---
    text = visual.TextStim(win=win, name='text',
        text='实验说明\n\n欢迎参加本次实验\n在接下来的任务中：\n\n1、您将看到一段人物表现出痛苦的视频\n2、请评价您感知的疼痛程度\n3、请决定您愿意提供多少帮助\n\n注意： 您的帮助意愿越高，可能需要等待的时间越长\n但人物的痛苦也会得到更有效的缓解\n接下来是练习部分，练习部分结束后会进入正式实验\n\n准备好后，请按 [空格键] 开始实验',
        font='Arial',
        units='pix', pos=(0, 0), draggable=False, height=70.0, wrapWidth=2000.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_5
    # 1. 强制显示鼠标
    win.mouseVisible = True 
    
    #初始化target_video，防止视频读取问题
    prac_movie = visual.MovieStim(win=win, name='prac_movie')
    target_video = visual.MovieStim(win=win, name='target_video')
    
    # 强制声明变量名，防止报错
    prac_feedback = visual.MovieStim(win=win, name='prac_feedback')
    feedback_player = visual.MovieStim(win=win, name='feedback_player')
    
    # 2. 全局退出键 (q)
    event.globalKeys.add(key='q', func=core.quit, name='shutdown')
    
    # 3. 初始化所有跨 Routine 变量，防止报错
    wait_duration = 0
    next_vid = ""
    prac_next_vid = ""
    
    # --- Initialize components for Routine "prac_stim" ---
    prac_polygon = visual.ShapeStim(
        win=win, name='prac_polygon', vertices='cross',units='pix', 
        size=(100, 100),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    prac_movie = visual.MovieStim(
        win, name='prac_movie',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=True,
        pos=(0, 0), size=(3072, 1920), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    prac_pain_text = visual.TextStim(win=win, name='prac_pain_text',
        text='请评价您感知的疼痛强度',
        font='Arial',
        units='pix', pos=(-150, 350), draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_help_text = visual.TextStim(win=win, name='prac_help_text',
        text='您愿意提供多少程度的帮助',
        font='Arial',
        units='pix', pos=(-150, -150), draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    prac_help_slider = visual.Slider(win=win, name='prac_help_slider',
        startValue=None, size=(1000, 50), pos=(0, -250), units='pix',
        labels=("不帮助", "最大帮助"), ticks=(0, 100), granularity=0.0,
        style='slider', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='SimHei', labelHeight=50.0,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    prac_pain_slider = visual.Slider(win=win, name='prac_pain_slider',
        startValue=None, size=(1000, 50), pos=(0, 250), units='pix',
        labels=("无痛", "极大痛苦"), ticks=(0, 100), granularity=0.0,
        style='slider', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='SimHei', labelHeight=50.0,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    
    # --- Initialize components for Routine "prac_waiting" ---
    prac_waiting_text = visual.TextStim(win=win, name='prac_waiting_text',
        text='',
        font='Arial',
        pos=(-200, 0), draggable=False, height=70.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "prac_trial_feedback" ---
    prac_feedback = visual.MovieStim(
        win, name='prac_feedback',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(3072, 1920), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    
    # --- Initialize components for Routine "text_routine" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='练习阶段已结束, 接下来的任务将计入正式实验数据,\n请保持专注\n按下[空格键]开始正式实验',
        font='Arial',
        pos=(-150, 0), draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "trial_stim" ---
    Pre_fixation = visual.ShapeStim(
        win=win, name='Pre_fixation', vertices='cross',units='pix', 
        size=(100, 100),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    target_video = visual.MovieStim(
        win, name='target_video',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=True,
        pos=(0, 0), size=(3072, 1920), units='pix',
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    pain_text = visual.TextStim(win=win, name='pain_text',
        text='请评价您感知的疼痛强度',
        font='Arial',
        units='pix', pos=(-150, 350), draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    help_text = visual.TextStim(win=win, name='help_text',
        text='您愿意提供多少程度的帮助',
        font='Arial',
        units='pix', pos=(-150, -150), draggable=False, height=60.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    help_slider = visual.Slider(win=win, name='help_slider',
        startValue=None, size=(1000, 50), pos=(0, -250), units='pix',
        labels=("不帮助", "最大帮助"), ticks=(0, 100), granularity=0.0,
        style='slider', styleTweaks=[], opacity=None,
        labelColor=(1.0000, 1.0000, 1.0000), markerColor='Red', lineColor='White', colorSpace='rgb',
        font='SimHei', labelHeight=50.0,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    pain_slider = visual.Slider(win=win, name='pain_slider',
        startValue=None, size=(1000, 50), pos=(0, 250), units='pix',
        labels=("无痛", "极大痛苦"), ticks=(0, 100), granularity=0.0,
        style='slider', styleTweaks=[], opacity=None,
        labelColor=(1.0000, 1.0000, 1.0000), markerColor='Red', lineColor='White', colorSpace='rgb',
        font='SimHei', labelHeight=50.0,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    
    # --- Initialize components for Routine "waiting_period" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(-200, 0), draggable=False, height=70.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "trial_feedback" ---
    feedback_player = visual.MovieStim(
        win, name='feedback_player',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=True,
        pos=(0, 0), size=(3072, 1920), units='pix',
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instr" ---
    # create an object to store info about Routine instr
    instr = data.Routine(
        name='instr',
        components=[text, key_resp],
    )
    instr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for instr
    instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr.tStart = globalClock.getTime(format='float')
    instr.status = STARTED
    thisExp.addData('instr.started', instr.tStart)
    instr.maxDuration = None
    # keep track of which components have finished
    instrComponents = instr.components
    for thisComponent in instr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr" ---
    thisExp.currentRoutine = instr
    instr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instr.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instr.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr" ---
    for thisComponent in instr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr
    instr.tStop = globalClock.getTime(format='float')
    instr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr.stopped', instr.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice_conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_loop in practice_loop:
        practice_loop.status = STARTED
        if hasattr(thisPractice_loop, 'status'):
            thisPractice_loop.status = STARTED
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "prac_stim" ---
        # create an object to store info about Routine prac_stim
        prac_stim = data.Routine(
            name='prac_stim',
            components=[prac_polygon, prac_movie, prac_pain_text, prac_help_text, prac_help_slider, prac_pain_slider],
        )
        prac_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_movie.setMovie(ORIGINAL_PATH)
        prac_help_slider.reset()
        prac_pain_slider.reset()
        # Run 'Begin Routine' code from prac_stim_code
        mouse = event.Mouse(visible=True, win=win)
        # store start times for prac_stim
        prac_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_stim.tStart = globalClock.getTime(format='float')
        prac_stim.status = STARTED
        thisExp.addData('prac_stim.started', prac_stim.tStart)
        prac_stim.maxDuration = None
        # keep track of which components have finished
        prac_stimComponents = prac_stim.components
        for thisComponent in prac_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_stim" ---
        thisExp.currentRoutine = prac_stim
        prac_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_loop, 'status') and thisPractice_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_polygon* updates
            
            # if prac_polygon is starting this frame...
            if prac_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_polygon.frameNStart = frameN  # exact frame index
                prac_polygon.tStart = t  # local t and not account for scr refresh
                prac_polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_polygon.started')
                # update status
                prac_polygon.status = STARTED
                prac_polygon.setAutoDraw(True)
            
            # if prac_polygon is active this frame...
            if prac_polygon.status == STARTED:
                # update params
                pass
            
            # if prac_polygon is stopping this frame...
            if prac_polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_polygon.tStop = t  # not accounting for scr refresh
                    prac_polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_polygon.stopped')
                    # update status
                    prac_polygon.status = FINISHED
                    prac_polygon.setAutoDraw(False)
            
            # *prac_movie* updates
            
            # if prac_movie is starting this frame...
            if prac_movie.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                prac_movie.frameNStart = frameN  # exact frame index
                prac_movie.tStart = t  # local t and not account for scr refresh
                prac_movie.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_movie, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_movie.started')
                # update status
                prac_movie.status = STARTED
                prac_movie.setAutoDraw(True)
                prac_movie.play()
            
            # if prac_movie is stopping this frame...
            if prac_movie.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_movie.tStartRefresh + 3.2-frameTolerance or prac_movie.isFinished:
                    # keep track of stop time/frame for later
                    prac_movie.tStop = t  # not accounting for scr refresh
                    prac_movie.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_movie.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_movie.stopped')
                    # update status
                    prac_movie.status = FINISHED
                    prac_movie.setAutoDraw(False)
            
            # *prac_pain_text* updates
            
            # if prac_pain_text is starting this frame...
            if prac_pain_text.status == NOT_STARTED and prac_movie.status == FINISHED:
                # keep track of start time/frame for later
                prac_pain_text.frameNStart = frameN  # exact frame index
                prac_pain_text.tStart = t  # local t and not account for scr refresh
                prac_pain_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_pain_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_pain_text.started')
                # update status
                prac_pain_text.status = STARTED
                prac_pain_text.setAutoDraw(True)
            
            # if prac_pain_text is active this frame...
            if prac_pain_text.status == STARTED:
                # update params
                pass
            
            # *prac_help_text* updates
            
            # if prac_help_text is starting this frame...
            if prac_help_text.status == NOT_STARTED and prac_movie.status == FINISHED:
                # keep track of start time/frame for later
                prac_help_text.frameNStart = frameN  # exact frame index
                prac_help_text.tStart = t  # local t and not account for scr refresh
                prac_help_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_help_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_help_text.started')
                # update status
                prac_help_text.status = STARTED
                prac_help_text.setAutoDraw(True)
            
            # if prac_help_text is active this frame...
            if prac_help_text.status == STARTED:
                # update params
                pass
            
            # *prac_help_slider* updates
            
            # if prac_help_slider is starting this frame...
            if prac_help_slider.status == NOT_STARTED and prac_movie.status == FINISHED:
                # keep track of start time/frame for later
                prac_help_slider.frameNStart = frameN  # exact frame index
                prac_help_slider.tStart = t  # local t and not account for scr refresh
                prac_help_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_help_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_help_slider.started')
                # update status
                prac_help_slider.status = STARTED
                prac_help_slider.setAutoDraw(True)
            
            # if prac_help_slider is active this frame...
            if prac_help_slider.status == STARTED:
                # update params
                pass
            
            # *prac_pain_slider* updates
            
            # if prac_pain_slider is starting this frame...
            if prac_pain_slider.status == NOT_STARTED and prac_movie.status == FINISHED:
                # keep track of start time/frame for later
                prac_pain_slider.frameNStart = frameN  # exact frame index
                prac_pain_slider.tStart = t  # local t and not account for scr refresh
                prac_pain_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_pain_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_pain_slider.started')
                # update status
                prac_pain_slider.status = STARTED
                prac_pain_slider.setAutoDraw(True)
            
            # if prac_pain_slider is active this frame...
            if prac_pain_slider.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from prac_stim_code
            # --- 1. 逻辑跳转 ---
            keys = event.getKeys(keyList=['space'])
            if 'space' in keys and t >= 3.0:
                # 获取评分
                prac_r1 = prac_pain_slider.getRating()
                prac_r2 = prac_help_slider.getRating()
                
                if prac_r1 is not None and prac_r2 is not None:
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_stim,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                prac_stim.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if prac_stim.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in prac_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_stim" ---
        for thisComponent in prac_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_stim
        prac_stim.tStop = globalClock.getTime(format='float')
        prac_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_stim.stopped', prac_stim.tStop)
        prac_movie.setAutoDraw(False)
        prac_movie.stop()  # ensure movie has stopped at end of Routine
        practice_loop.addData('prac_help_slider.response', prac_help_slider.getRating())
        practice_loop.addData('prac_help_slider.rt', prac_help_slider.getRT())
        practice_loop.addData('prac_pain_slider.response', prac_pain_slider.getRating())
        practice_loop.addData('prac_pain_slider.rt', prac_pain_slider.getRT())
        # Run 'End Routine' code from prac_stim_code
        # 1. 获取参与者在帮助意愿量表上的得分
        # help_slider 是你给量表起的名字
        prac_score = prac_help_slider.getRating()
        
        # 2. 根据得分区间，从 Excel 的列中选择对应的路径
        # 这些变量名（nodiff_path等）必须和 Excel 的表头完全一致 [cite: 31, 263]
        if prac_score is not None:
            if prac_score <= 32:
                prac_next_vid = NODIFFERENCE_PATH
            elif score <= 65:
                prac_next_vid = MODERATE_PATH
            else:
                prac_next_vid = RELIEF_PATH
        else:
            # 防止极端情况（没点就跳过了），默认给个不缓解
            prac_next_vid = NODIFFERENCE_PATH
        
        # 3. 把这个路径存入数据文件，方便你以后分析 [cite: 491]
        thisExp.addData('prac_feedback_video_used', prac_next_vid)
        
        # 每一轮结束都强制让播放器“归零”，释放内存
        prac_movie.stop() 
        # 如果你的反馈视频组件叫 feedback_video，也加上：
        # feedback_video.stop()
        # the Routine "prac_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_waiting" ---
        # create an object to store info about Routine prac_waiting
        prac_waiting = data.Routine(
            name='prac_waiting',
            components=[prac_waiting_text],
        )
        prac_waiting.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prac_waiting_code
        # 拿到刚才的分数并转化为秒
        prac_val = prac_help_slider.getRating()
        if prac_val is None: prac_val = 0
        prac_wait_duration = (prac_val / 100.0) * 15
        prac_wait_timer = core.Clock()
        prac_waiting_text.setText(u"已接受到帮助，请您等待"+ str(round(wait_duration, 1)) + u"秒")
        # store start times for prac_waiting
        prac_waiting.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_waiting.tStart = globalClock.getTime(format='float')
        prac_waiting.status = STARTED
        thisExp.addData('prac_waiting.started', prac_waiting.tStart)
        prac_waiting.maxDuration = None
        # keep track of which components have finished
        prac_waitingComponents = prac_waiting.components
        for thisComponent in prac_waiting.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_waiting" ---
        thisExp.currentRoutine = prac_waiting
        prac_waiting.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_loop, 'status') and thisPractice_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from prac_waiting_code
            # 强行停留，直到时间走完
            if prac_wait_timer.getTime() >= wait_duration:
                continueRoutine = False
            
            # *prac_waiting_text* updates
            
            # if prac_waiting_text is active this frame...
            if prac_waiting_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_waiting,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                prac_waiting.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if prac_waiting.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in prac_waiting.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_waiting" ---
        for thisComponent in prac_waiting.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_waiting
        prac_waiting.tStop = globalClock.getTime(format='float')
        prac_waiting.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_waiting.stopped', prac_waiting.tStop)
        # the Routine "prac_waiting" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_trial_feedback" ---
        # create an object to store info about Routine prac_trial_feedback
        prac_trial_feedback = data.Routine(
            name='prac_trial_feedback',
            components=[prac_feedback],
        )
        prac_trial_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_feedback.setMovie(prac_next_vid)
        # Run 'Begin Routine' code from prac_feedback_code
        # 如果因为某种意外 next_vid 没有拿到底稿，直接跳过这个反馈环节，防止程序卡死
        if prac_next_vid == "" or prac_next_vid is None:
            continueRoutine = False
        # store start times for prac_trial_feedback
        prac_trial_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_trial_feedback.tStart = globalClock.getTime(format='float')
        prac_trial_feedback.status = STARTED
        thisExp.addData('prac_trial_feedback.started', prac_trial_feedback.tStart)
        prac_trial_feedback.maxDuration = None
        # keep track of which components have finished
        prac_trial_feedbackComponents = prac_trial_feedback.components
        for thisComponent in prac_trial_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_trial_feedback" ---
        thisExp.currentRoutine = prac_trial_feedback
        prac_trial_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_loop, 'status') and thisPractice_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_feedback* updates
            
            # if prac_feedback is starting this frame...
            if prac_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_feedback.frameNStart = frameN  # exact frame index
                prac_feedback.tStart = t  # local t and not account for scr refresh
                prac_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_feedback.started')
                # update status
                prac_feedback.status = STARTED
                prac_feedback.setAutoDraw(True)
                prac_feedback.play()
            
            # if prac_feedback is stopping this frame...
            if prac_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_feedback.tStartRefresh + 1.0-frameTolerance or prac_feedback.isFinished:
                    # keep track of stop time/frame for later
                    prac_feedback.tStop = t  # not accounting for scr refresh
                    prac_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_feedback.stopped')
                    # update status
                    prac_feedback.status = FINISHED
                    prac_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_trial_feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                prac_trial_feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if prac_trial_feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in prac_trial_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_trial_feedback" ---
        for thisComponent in prac_trial_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_trial_feedback
        prac_trial_feedback.tStop = globalClock.getTime(format='float')
        prac_trial_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_trial_feedback.stopped', prac_trial_feedback.tStop)
        prac_feedback.setAutoDraw(False)
        prac_feedback.stop()  # ensure movie has stopped at end of Routine
        # Run 'End Routine' code from prac_feedback_code
        # 记录这一个试次最终呈现给受试者的反馈视频路径
        thisExp.addData('prac_final_feedback_video', prac_next_vid)
        # 彻底杀掉当前播放器进程，强迫它在下一轮重新开始，而不是在内存里堆积
        prac_feedback.status = FINISHED
        prac_feedback.stop()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_trial_feedback.maxDurationReached:
            routineTimer.addTime(-prac_trial_feedback.maxDuration)
        elif prac_trial_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPractice_loop as finished
        if hasattr(thisPractice_loop, 'status'):
            thisPractice_loop.status = FINISHED
        # if awaiting a pause, pause now
        if practice_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_loop'
    practice_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "text_routine" ---
    # create an object to store info about Routine text_routine
    text_routine = data.Routine(
        name='text_routine',
        components=[text_3, key_resp_2],
    )
    text_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for text_routine
    text_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    text_routine.tStart = globalClock.getTime(format='float')
    text_routine.status = STARTED
    thisExp.addData('text_routine.started', text_routine.tStart)
    text_routine.maxDuration = None
    # keep track of which components have finished
    text_routineComponents = text_routine.components
    for thisComponent in text_routine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text_routine" ---
    thisExp.currentRoutine = text_routine
    text_routine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=text_routine,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            text_routine.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if text_routine.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in text_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text_routine" ---
    for thisComponent in text_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for text_routine
    text_routine.tStop = globalClock.getTime(format='float')
    text_routine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('text_routine.stopped', text_routine.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "text_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial_stim" ---
        # create an object to store info about Routine trial_stim
        trial_stim = data.Routine(
            name='trial_stim',
            components=[Pre_fixation, target_video, pain_text, help_text, help_slider, pain_slider],
        )
        trial_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        target_video.setMovie(ORIGINAL_PATH)
        help_slider.reset()
        pain_slider.reset()
        # Run 'Begin Routine' code from code
        mouse = event.Mouse(visible=True, win=win)
        # store start times for trial_stim
        trial_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_stim.tStart = globalClock.getTime(format='float')
        trial_stim.status = STARTED
        thisExp.addData('trial_stim.started', trial_stim.tStart)
        trial_stim.maxDuration = None
        # keep track of which components have finished
        trial_stimComponents = trial_stim.components
        for thisComponent in trial_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_stim" ---
        thisExp.currentRoutine = trial_stim
        trial_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Pre_fixation* updates
            
            # if Pre_fixation is starting this frame...
            if Pre_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pre_fixation.frameNStart = frameN  # exact frame index
                Pre_fixation.tStart = t  # local t and not account for scr refresh
                Pre_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pre_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Pre_fixation.started')
                # update status
                Pre_fixation.status = STARTED
                Pre_fixation.setAutoDraw(True)
            
            # if Pre_fixation is active this frame...
            if Pre_fixation.status == STARTED:
                # update params
                pass
            
            # if Pre_fixation is stopping this frame...
            if Pre_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pre_fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Pre_fixation.tStop = t  # not accounting for scr refresh
                    Pre_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    Pre_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Pre_fixation.stopped')
                    # update status
                    Pre_fixation.status = FINISHED
                    Pre_fixation.setAutoDraw(False)
            
            # *target_video* updates
            
            # if target_video is starting this frame...
            if target_video.status == NOT_STARTED and tThisFlip >= 1.2-frameTolerance:
                # keep track of start time/frame for later
                target_video.frameNStart = frameN  # exact frame index
                target_video.tStart = t  # local t and not account for scr refresh
                target_video.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_video, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_video.started')
                # update status
                target_video.status = STARTED
                target_video.setAutoDraw(True)
                target_video.play()
            
            # if target_video is stopping this frame...
            if target_video.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_video.tStartRefresh + 2.0-frameTolerance or target_video.isFinished:
                    # keep track of stop time/frame for later
                    target_video.tStop = t  # not accounting for scr refresh
                    target_video.tStopRefresh = tThisFlipGlobal  # on global time
                    target_video.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_video.stopped')
                    # update status
                    target_video.status = FINISHED
                    target_video.setAutoDraw(False)
            
            # *pain_text* updates
            
            # if pain_text is starting this frame...
            if pain_text.status == NOT_STARTED and target_video.status == FINISHED:
                # keep track of start time/frame for later
                pain_text.frameNStart = frameN  # exact frame index
                pain_text.tStart = t  # local t and not account for scr refresh
                pain_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_text.started')
                # update status
                pain_text.status = STARTED
                pain_text.setAutoDraw(True)
            
            # if pain_text is active this frame...
            if pain_text.status == STARTED:
                # update params
                pass
            
            # *help_text* updates
            
            # if help_text is starting this frame...
            if help_text.status == NOT_STARTED and target_video.status == FINISHED:
                # keep track of start time/frame for later
                help_text.frameNStart = frameN  # exact frame index
                help_text.tStart = t  # local t and not account for scr refresh
                help_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(help_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'help_text.started')
                # update status
                help_text.status = STARTED
                help_text.setAutoDraw(True)
            
            # if help_text is active this frame...
            if help_text.status == STARTED:
                # update params
                pass
            
            # *help_slider* updates
            
            # if help_slider is starting this frame...
            if help_slider.status == NOT_STARTED and target_video.status == FINISHED:
                # keep track of start time/frame for later
                help_slider.frameNStart = frameN  # exact frame index
                help_slider.tStart = t  # local t and not account for scr refresh
                help_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(help_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'help_slider.started')
                # update status
                help_slider.status = STARTED
                help_slider.setAutoDraw(True)
            
            # if help_slider is active this frame...
            if help_slider.status == STARTED:
                # update params
                pass
            
            # *pain_slider* updates
            
            # if pain_slider is starting this frame...
            if pain_slider.status == NOT_STARTED and target_video.status == FINISHED:
                # keep track of start time/frame for later
                pain_slider.frameNStart = frameN  # exact frame index
                pain_slider.tStart = t  # local t and not account for scr refresh
                pain_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_slider.started')
                # update status
                pain_slider.status = STARTED
                pain_slider.setAutoDraw(True)
            
            # if pain_slider is active this frame...
            if pain_slider.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code
            # --- 1. 逻辑跳转 ---
            keys = event.getKeys(keyList=['space'])
            if 'space' in keys and t >= 3.0:
                # 获取评分
                r1 = pain_slider.getRating()
                r2 = help_slider.getRating()
                
                if r1 is not None and r2 is not None:
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_stim,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_stim.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_stim.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_stim" ---
        for thisComponent in trial_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_stim
        trial_stim.tStop = globalClock.getTime(format='float')
        trial_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_stim.stopped', trial_stim.tStop)
        target_video.setAutoDraw(False)
        target_video.stop()  # ensure movie has stopped at end of Routine
        trials.addData('help_slider.response', help_slider.getRating())
        trials.addData('help_slider.rt', help_slider.getRT())
        trials.addData('help_slider.history', help_slider.getHistory())
        trials.addData('pain_slider.response', pain_slider.getRating())
        trials.addData('pain_slider.rt', pain_slider.getRT())
        # Run 'End Routine' code from code
        # 1. 获取参与者在帮助意愿量表上的得分
        # help_slider 是你给量表起的名字
        score = help_slider.getRating()
        
        # 2. 根据得分区间，从 Excel 的列中选择对应的路径
        # 这些变量名（nodiff_path等）必须和 Excel 的表头完全一致 [cite: 31, 263]
        if score is not None:
            if score <= 32:
                next_vid = NODIFFERENCE_PATH
            elif score <= 65:
                next_vid = MODERATE_PATH
            else:
                next_vid = RELIEF_PATH
        else:
            # 防止极端情况（没点就跳过了），默认给个不缓解
            next_vid = NODIFFERENCE_PATH
        
        # 3. 把这个路径存入数据文件，方便你以后分析 [cite: 491]
        thisExp.addData('feedback_video_used', next_vid)
        
        # 每一轮结束都强制让播放器“归零”，释放内存
        target_video.stop() 
        # 如果你的反馈视频组件叫 feedback_video，也加上：
        # feedback_video.stop()
        # the Routine "trial_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "waiting_period" ---
        # create an object to store info about Routine waiting_period
        waiting_period = data.Routine(
            name='waiting_period',
            components=[text_2],
        )
        waiting_period.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_6
        # 拿到刚才的分数并转化为秒
        val = help_slider.getRating()
        if val is None: val = 0
        wait_duration = (val / 100.0) * 15
        wait_timer = core.Clock()
        text_2.setText(u"已接受到帮助，请您等待"+ str(round(wait_duration, 1)) + u"秒")
        # store start times for waiting_period
        waiting_period.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        waiting_period.tStart = globalClock.getTime(format='float')
        waiting_period.status = STARTED
        thisExp.addData('waiting_period.started', waiting_period.tStart)
        waiting_period.maxDuration = None
        # keep track of which components have finished
        waiting_periodComponents = waiting_period.components
        for thisComponent in waiting_period.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "waiting_period" ---
        thisExp.currentRoutine = waiting_period
        waiting_period.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_6
            # 强行停留，直到时间走完
            if wait_timer.getTime() >= wait_duration:
                continueRoutine = False
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=waiting_period,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                waiting_period.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if waiting_period.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in waiting_period.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "waiting_period" ---
        for thisComponent in waiting_period.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for waiting_period
        waiting_period.tStop = globalClock.getTime(format='float')
        waiting_period.tStopRefresh = tThisFlipGlobal
        thisExp.addData('waiting_period.stopped', waiting_period.tStop)
        # the Routine "waiting_period" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_feedback" ---
        # create an object to store info about Routine trial_feedback
        trial_feedback = data.Routine(
            name='trial_feedback',
            components=[feedback_player],
        )
        trial_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_player.setMovie(next_vid)
        # Run 'Begin Routine' code from code_3
        # 如果因为某种意外 next_vid 没有拿到底稿，直接跳过这个反馈环节，防止程序卡死
        if next_vid == "" or next_vid is None:
            continueRoutine = False
        # store start times for trial_feedback
        trial_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_feedback.tStart = globalClock.getTime(format='float')
        trial_feedback.status = STARTED
        thisExp.addData('trial_feedback.started', trial_feedback.tStart)
        trial_feedback.maxDuration = None
        # keep track of which components have finished
        trial_feedbackComponents = trial_feedback.components
        for thisComponent in trial_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_feedback" ---
        thisExp.currentRoutine = trial_feedback
        trial_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_player* updates
            
            # if feedback_player is starting this frame...
            if feedback_player.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_player.frameNStart = frameN  # exact frame index
                feedback_player.tStart = t  # local t and not account for scr refresh
                feedback_player.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_player, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_player.started')
                # update status
                feedback_player.status = STARTED
                feedback_player.setAutoDraw(True)
                feedback_player.play()
            
            # if feedback_player is stopping this frame...
            if feedback_player.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_player.tStartRefresh + 1.0-frameTolerance or feedback_player.isFinished:
                    # keep track of stop time/frame for later
                    feedback_player.tStop = t  # not accounting for scr refresh
                    feedback_player.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_player.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_player.stopped')
                    # update status
                    feedback_player.status = FINISHED
                    feedback_player.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_feedback" ---
        for thisComponent in trial_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_feedback
        trial_feedback.tStop = globalClock.getTime(format='float')
        trial_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_feedback.stopped', trial_feedback.tStop)
        feedback_player.setAutoDraw(False)
        feedback_player.stop()  # ensure movie has stopped at end of Routine
        # Run 'End Routine' code from code_3
        # 记录这一个试次最终呈现给受试者的反馈视频路径
        thisExp.addData('final_feedback_video', next_vid)
        # 彻底杀掉当前播放器进程，强迫它在下一轮重新开始，而不是在内存里堆积
        feedback_player.status = FINISHED
        feedback_player.stop()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_feedback.maxDurationReached:
            routineTimer.addTime(-trial_feedback.maxDuration)
        elif trial_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
