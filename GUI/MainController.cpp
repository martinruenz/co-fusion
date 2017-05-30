/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "../Core/Utils/Macros.h"
#include "MainController.h"
#include "Tools/KlgLogReader.h"
#include "Tools/LiveLogReader.h"
#include "Tools/ImageLogReader.h"

#include <boost/filesystem.hpp>

/*
 * Parameters:

    -run    Run dataset immediately (otherwise start paused).
    -q      Quit when finished a log.
    -cal    Loads a camera calibration file specified as fx fy cx cy.
    -p      Loads ground truth poses to use instead of estimated pose.
    -d      Cutoff distance for depth processing (default 5m).
    -i      Relative ICP/RGB tracking weight (default 10).
    -or     Outlier rejection strength (default 3).
    -ie     Local loop closure residual threshold (default 5e-05).
    -ic     Local loop closure inlier threshold (default 35000).
    -cv     Local loop closure covariance threshold (default 1e-05).
    -pt     Global loop closure photometric threshold (default 115).
    -ft     Fern encoding threshold (default 0.3095).
    -t      Time window length (default 200).
    -s      Frames to skip at start of log.
    -e      Cut off frame of log.
    -f      Flip RGB/BGR.
    -icl    Enable this if using the ICL-NUIM dataset (flips normals to account for negative focal length on that data).
    -o      Open loop mode.
    -rl     Enable relocalisation.
    -fs     Frame skip if processing a log to simulate real-time.
    -fo     Fast odometry (single level pyramid).
    -nso    Disables SO(3) pre-alignment in tracking.
    -r      Rewind and loop log forever.
    -ftf    Do frame-to-frame RGB tracking.
    -sc     Showcase mode (minimal GUI).
    -vxp    Use visionx point cloud reader. Provide the name of the Provider.
    -vxf    Use together with visionx point cloud reader option. Provide the name for the file.

    -static        Disable multi-model fusion.
    -confO         Initial surfel confidence threshold for objects (default 0.01).
    -confG         Initial surfel confidence threshold for scene (default 10.00).
    -segMinNew     Min size of new object segments (relative to image size)
    -segMaxNew     Max size of new object segments (relative to image size)
    -offset        Offset between creating models
    -keep          Keep all models (even bad, deactivated)

    -l             Processes a log-file (*.klg).
    -dir           Processes a log-directory (Default: Color####.png + Depth####.exr [+ Mask####.png])
    -depthdir      Separate depth directory (==dir if not provided)
    -maskdir       Separate mask directory (==dir if not provided)
    -exportdir     Export results to this directory, otherwise not exported
    -basedir       Treat the above paths relative to this one (like depthdir = basedir + depthdir, default "")
    -colorprefix   Specify prefix of color files (=="" or =="Color" if not provided)
    -depthprefix   Specify prefix of depth files (=="" or =="Depth" if not provided)
    -maskprefix    Specify prefix of mask files (=="" or =="Mask" if not provided)
    -indexW        Number of digits of the indexes (==4 if not provided)
    -nm            Ignore Mask####.png images as soon as the provided frame was reached.
    -es            Export segmentation
    -ev            Export viewport images
    -el            Export label images
    -em            Export models (point-cloud)
    -en            Export normal images
    -ep            Export poses after finishing run (just before quitting if '-q')

    Examples:
    -basedir /mnt/path/to/my/dataset -dir color_dir -depthdir depth_dir -maskdir mask_dir -depthprefix Depth -colorprefix Color
    -maskprefix Mask -cal calibration.txt
 */

MainController::MainController(int argc, char* argv[])
    : good(true), coFusion(0), gui(0), groundTruthOdometry(0), logReader(nullptr), framesToSkip(0), resetButton(false), resizeStream(0) {
  std::string empty;
  float tmpFloat;
  iclnuim = Parse::get().arg(argc, argv, "-icl", empty) > -1;

  std::string baseDir;
  Parse::get().arg(argc, argv, "-basedir", baseDir);
  if (baseDir.length()) baseDir += '/';

  std::string calibrationFile;
  Parse::get().arg(argc, argv, "-cal", calibrationFile);
  calibrationFile = baseDir + calibrationFile;

  // Asus is default camera (might change later)
  Resolution::setResolution(640, 480);
  Intrinsics::setIntrinics(528, 528, 320, 240);

  if (calibrationFile.length()) loadCalibration(calibrationFile);

  bool logReaderReady = false;

  Parse::get().arg(argc, argv, "-l", logFile);
  if (logFile.length()) {
    logReader = std::make_unique<KlgLogReader>(logFile, Parse::get().arg(argc, argv, "-f", empty) > -1);
    logReaderReady = true;
  }

  if (!logReaderReady) {
    Parse::get().arg(argc, argv, "-dir", logFile);
    if (logFile.length()) {
      logFile += '/';  // "colorDir"
      std::string depthDir, maskDir, depthPrefix, colorPrefix, maskPrefix;
      Parse::get().arg(argc, argv, "-depthdir", depthDir);
      Parse::get().arg(argc, argv, "-maskdir", maskDir);
      Parse::get().arg(argc, argv, "-colorprefix", colorPrefix);
      Parse::get().arg(argc, argv, "-depthprefix", depthPrefix);
      Parse::get().arg(argc, argv, "-maskprefix", maskPrefix);
      if (depthDir.length())
        depthDir += '/';
      else
        depthDir = logFile;
      if (maskDir.length())
        maskDir += '/';
      else
        maskDir = logFile;
      int indexW = -1;
      ImageLogReader* imageLogReader = new ImageLogReader(baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
                                                          Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW : 4, colorPrefix,
                                                          depthPrefix, maskPrefix, Parse::get().arg(argc, argv, "-f", empty) > -1);

      // How many masks?
      int maxMasks = -1;
      if (Parse::get().arg(argc, argv, "-nm", maxMasks) > -1) {
        if (maxMasks >= 0)
          imageLogReader->setMaxMasks(maxMasks);
        else
          imageLogReader->ignoreMask();
      }

      logReader = std::unique_ptr<LogReader>(imageLogReader);
      logReaderReady = true;
    }
  }

  if (!logReaderReady) {
    logReader = std::make_unique<LiveLogReader>(logFile, Parse::get().arg(argc, argv, "-f", empty) > -1);
    good = ((LiveLogReader*)logReader.get())->asus->ok();
  }

  if (logReader->hasIntrinsics() && !calibrationFile.length()) loadCalibration(logReader->getIntinsicsFile());

  if (Parse::get().arg(argc, argv, "-p", poseFile) > 0) {
    groundTruthOdometry = new GroundTruthOdometry(poseFile);
  }

  confObjectInit = 0.01f;
  confGlobalInit = 10.0f;
  icpErrThresh = 5e-05;
  covThresh = 1e-05;
  photoThresh = 115;
  fernThresh = 0.3095f;

  timeDelta = 200;  // Ignored, since openLoop
  icpCountThresh = 35000;
  start = 1;
  so3 = !(Parse::get().arg(argc, argv, "-nso", empty) > -1);
  end = std::numeric_limits<unsigned short>::max();  // Funny bound, since we predict times in this format really!

  Parse::get().arg(argc, argv, "-confG", confGlobalInit);
  Parse::get().arg(argc, argv, "-confO", confObjectInit);
  Parse::get().arg(argc, argv, "-ie", icpErrThresh);
  Parse::get().arg(argc, argv, "-cv", covThresh);
  Parse::get().arg(argc, argv, "-pt", photoThresh);
  Parse::get().arg(argc, argv, "-ft", fernThresh);
  Parse::get().arg(argc, argv, "-t", timeDelta);
  Parse::get().arg(argc, argv, "-ic", icpCountThresh);
  Parse::get().arg(argc, argv, "-s", start);
  Parse::get().arg(argc, argv, "-e", end);

  logReader->flipColors = Parse::get().arg(argc, argv, "-f", empty) > -1;

  openLoop = true;  // FIXME //!groundTruthOdometry && (Parse::get().arg(argc, argv, "-o", empty) > -1);
  reloc = Parse::get().arg(argc, argv, "-rl", empty) > -1;
  frameskip = Parse::get().arg(argc, argv, "-fs", empty) > -1;
  quit = Parse::get().arg(argc, argv, "-q", empty) > -1;
  fastOdom = Parse::get().arg(argc, argv, "-fo", empty) > -1;
  rewind = Parse::get().arg(argc, argv, "-r", empty) > -1;
  frameToFrameRGB = Parse::get().arg(argc, argv, "-ftf", empty) > -1;
  exportSegmentation = Parse::get().arg(argc, argv, "-es", empty) > -1;
  exportViewport = Parse::get().arg(argc, argv, "-ev", empty) > -1;
  exportLabels = Parse::get().arg(argc, argv, "-el", empty) > -1;
  exportNormals = Parse::get().arg(argc, argv, "-en", empty) > -1;
  exportPoses = Parse::get().arg(argc, argv, "-ep", empty) > -1;
  exportModels = Parse::get().arg(argc, argv, "-em", empty) > -1;

  showcaseMode = Parse::get().arg(argc, argv, "-sc", empty) > -1;
  gui = new GUI(logFile.length() == 0, showcaseMode);

  if (Parse::get().arg(argc, argv, "-d", tmpFloat) > -1) gui->depthCutoff->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-i", tmpFloat) > -1) gui->icpWeight->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-or", tmpFloat) > -1) gui->outlierCoefficient->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-segMinNew", tmpFloat) > -1) gui->minRelSizeNew->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-segMaxNew", tmpFloat) > -1) gui->maxRelSizeNew->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-crfRGB", tmpFloat) > -1) gui->pairwiseRGBSTD->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-crfDepth", tmpFloat) > -1) gui->pairwiseDepthSTD->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-crfPos", tmpFloat) > -1) gui->pairwisePosSTD->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-crfAppearance", tmpFloat) > -1) gui->pairwiseAppearanceWeight->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-crfSmooth", tmpFloat) > -1) gui->pairwiseSmoothnessWeight->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-offset", tmpFloat) > -1) gui->modelSpawnOffset->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-thNew", tmpFloat) > -1) gui->thresholdNew->Ref().Set(tmpFloat);
  if (Parse::get().arg(argc, argv, "-k", tmpFloat) > -1) gui->unaryErrorK->Ref().Set(tmpFloat);

  gui->flipColors->Ref().Set(logReader->flipColors);
  gui->rgbOnly->Ref().Set(false);
  gui->enableMultiModel->Ref().Set(Parse::get().arg(argc, argv, "-static", empty) <= -1);
  gui->enableSmartDelete->Ref().Set(Parse::get().arg(argc, argv, "-keep", empty) <= -1);
  gui->pyramid->Ref().Set(true);
  gui->fastOdom->Ref().Set(fastOdom);
  // gui->confidenceThreshold->Ref().Set(confidence);
  gui->so3->Ref().Set(so3);
  gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);
  gui->pause->Ref().Set((Parse::get().arg(argc, argv, "-run", empty) <= -1));
  // gui->pause->Ref().Set(logFile.length());
  // gui->pause->Ref().Set(!showcaseMode);

  resizeStream = new GPUResize(Resolution::getInstance().width(), Resolution::getInstance().height(), Resolution::getInstance().width() / 2,
                               Resolution::getInstance().height() / 2);

  if (Parse::get().arg(argc, argv, "-exportdir", exportDir) > 0) {
    if (exportDir.length() == 0 || exportDir[0] != '/') exportDir = baseDir + exportDir;
  } else {
    exportDir = baseDir + logFile + "-export/";
  }
  exportDir += "/";

  // Create export dir if it doesn't exist
  boost::filesystem::path eDir(exportDir);
  boost::filesystem::create_directory(eDir);

  std::cout << "Initialised MainController. Frame resolution is set to: " << Resolution::getInstance().width() << "x"
            << Resolution::getInstance().height() << "\n"
                                                     "Exporting results to: "
            << exportDir << std::endl;
}

MainController::~MainController() {
  if (coFusion) {
    delete coFusion;
  }

  if (gui) {
    delete gui;
  }

  if (groundTruthOdometry) {
    delete groundTruthOdometry;
  }

  if (resizeStream) {
    delete resizeStream;
  }
}

void MainController::loadCalibration(const std::string& filename) {
  std::cout << "Loading camera parameters from file: " << filename << std::endl;

  std::ifstream file(filename);
  std::string line;

  CHECK_THROW(!file.eof());

  double fx, fy, cx, cy, w, h;

  std::getline(file, line);

  int n = sscanf(line.c_str(), "%lg %lg %lg %lg %lg %lg", &fx, &fy, &cx, &cy, &w, &h);

  if (n != 4 && n != 6)
    throw std::invalid_argument("Ooops, your calibration file should contain a single line with [fx fy cx cy] or [fx fy cx cy w h]");

  Intrinsics::setIntrinics(fx, fy, cx, cy);
  if (n == 6) Resolution::setResolution(w, h);
}

void MainController::launch() {
  while (good) {
    if (coFusion) {
      run();
    }

    if (coFusion == 0 || resetButton) {
      resetButton = false;

      if (coFusion) {
        delete coFusion;
        cudaCheckError();
      }

      coFusion =
          new CoFusion(openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta, icpCountThresh, icpErrThresh, covThresh, !openLoop,
                       iclnuim, reloc, photoThresh, confGlobalInit, confObjectInit, gui->depthCutoff->Get(), gui->icpWeight->Get(), fastOdom, fernThresh, so3, frameToFrameRGB,
                       gui->modelSpawnOffset->Get(), Model::MatchingType::Drost, exportDir, exportSegmentation);

      auto globalModel = coFusion->getBackgroundModel();
      gui->addModel(globalModel->getID(), globalModel->getConfidenceThreshold());

      coFusion->addNewModelListener(
          [this](std::shared_ptr<Model> model) { gui->addModel(model->getID(), model->getConfidenceThreshold()); });
      // eFusion->addNewModelListener([this](std::shared_ptr<Model> model){
      //    gui->addModel(model->getID(), model->getConfidenceThreshold());}
      //);
    } else {
      break;
    }
  }
}

void MainController::run() {
  while (!pangolin::ShouldQuit() && !((!logReader->hasMore()) && quit) && !(coFusion->getTick() == end && quit)) {
    if (!gui->pause->Get() || pangolin::Pushed(*gui->step)) {
      if ((logReader->hasMore() || rewind) && coFusion->getTick() < end) {
        TICK("LogRead");
        if (rewind) {
          if (!logReader->hasMore()) {
            logReader->getBack();
          } else {
            logReader->getNext();
          }

          if (logReader->rewound()) {
            logReader->currentFrame = 0;
          }
        } else {
          logReader->getNext();
        }
        TOCK("LogRead");

        if (coFusion->getTick() < start) {
          coFusion->setTick(start);
          logReader->fastForward(start);
        }

        float weightMultiplier = framesToSkip + 1;

        if (framesToSkip > 0) {
          coFusion->setTick(coFusion->getTick() + framesToSkip);
          logReader->fastForward(logReader->currentFrame + framesToSkip);
          framesToSkip = 0;
        }

        Eigen::Matrix4f* currentPose = 0;

        if (groundTruthOdometry) {
          currentPose = new Eigen::Matrix4f;
          currentPose->setIdentity();
          *currentPose = groundTruthOdometry->getIncrementalTransformation(logReader->getFrameData().timestamp);
        }

        if (coFusion->processFrame(logReader->getFrameData(), currentPose, weightMultiplier) && !showcaseMode) {
          gui->pause->Ref().Set(true);
        }

        if (exportLabels) {
          gui->saveColorImage(exportDir + "Labels" + std::to_string(coFusion->getTick() - 1));
          drawScene(DRAW_COLOR, DRAW_LABEL);
        }

        if (exportNormals) {
          gui->saveColorImage(exportDir + "Normals" + std::to_string(coFusion->getTick() - 1));
          drawScene(DRAW_NORMALS, DRAW_NORMALS);
        }

        if (exportViewport) {
          gui->saveColorImage(exportDir + "Viewport" + std::to_string(coFusion->getTick() - 1));
          // drawScene();
        }

        if (currentPose) {
          delete currentPose;
        }

        if (frameskip && Stopwatch::getInstance().getTimings().at("Run") > 1000.f / 30.f) {
          framesToSkip = int(Stopwatch::getInstance().getTimings().at("Run") / (1000.f / 30.f));
        }
      }
    } else if (pangolin::Pushed(*gui->skip)) {
      coFusion->setTick(coFusion->getTick() + 1);
      logReader->fastForward(logReader->currentFrame + 1);
    } else {
      coFusion->predict();

      // TODO Only if relevant setting changed (Deactivate when writing (debug/visualisation) images to hd
      if(logReader->getFrameData().timestamp) coFusion->performSegmentation(logReader->getFrameData());
    }

    TICK("GUI");

    std::stringstream stri;
    stri << coFusion->getModelToModel().lastICPCount;
    gui->trackInliers->Ref().Set(stri.str());

    std::stringstream stre;
    stre << (std::isnan(coFusion->getModelToModel().lastICPError) ? 0 : coFusion->getModelToModel().lastICPError);
    gui->trackRes->Ref().Set(stre.str());

    if (!gui->pause->Get()) {
      gui->resLog.Log((std::isnan(coFusion->getModelToModel().lastICPError) ? std::numeric_limits<float>::max()
                                                                            : coFusion->getModelToModel().lastICPError),
                      icpErrThresh);
      gui->inLog.Log(coFusion->getModelToModel().lastICPCount, icpCountThresh);
    }

    drawScene();

    // SET PARAMETERS / SETTINGS
    logReader->flipColors = gui->flipColors->Get();
    coFusion->setEnableMultipleModels(gui->enableMultiModel->Get());
    coFusion->setEnableSmartModelDelete(gui->enableSmartDelete->Get());
    coFusion->setRgbOnly(gui->rgbOnly->Get());
    coFusion->setPyramid(gui->pyramid->Get());
    coFusion->setFastOdom(gui->fastOdom->Get());
    coFusion->setDepthCutoff(gui->depthCutoff->Get());
    coFusion->setIcpWeight(gui->icpWeight->Get());
    coFusion->setOutlierCoefficient(gui->outlierCoefficient->Get());
    coFusion->setSo3(gui->so3->Get());
    coFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());

    coFusion->setModelSpawnOffset(gui->modelSpawnOffset->Get());
    coFusion->setModelDeactivateCount(gui->modelDeactivateCnt->Get());
    coFusion->setNewModelMinRelativeSize(gui->minRelSizeNew->Get());
    coFusion->setNewModelMaxRelativeSize(gui->maxRelSizeNew->Get());
    coFusion->setCrfPairwiseWeightAppearance(gui->pairwiseAppearanceWeight->Get());
    coFusion->setCrfPairwiseWeightSmoothness(gui->pairwiseSmoothnessWeight->Get());
    coFusion->setCrfPairwiseSigmaDepth(gui->pairwiseDepthSTD->Get());
    coFusion->setCrfPairwiseSigmaPosition(gui->pairwisePosSTD->Get());
    coFusion->setCrfPairwiseSigmaRGB(gui->pairwiseRGBSTD->Get());
    coFusion->setCrfThresholdNew(gui->thresholdNew->Get());
    coFusion->setCrfUnaryKError(gui->unaryErrorK->Get());
    coFusion->setCrfUnaryWeightError(gui->unaryErrorWeight->Get());
    coFusion->setCrfIteration(gui->crfIterations->Get());

    resetButton = pangolin::Pushed(*gui->reset);

    if (gui->autoSettings) {
      static bool last = gui->autoSettings->Get();

      if (gui->autoSettings->Get() != last) {
        last = gui->autoSettings->Get();
        // static_cast<LiveLogReader *>(logReader)->setAuto(last);
        logReader->setAuto(last);
      }
    }

    Stopwatch::getInstance().sendAll();

    if (resetButton) {
      break;
    }

    if (pangolin::Pushed(*gui->saveCloud)) coFusion->savePly();
    // if(pangolin::Pushed(*gui->saveDepth)) eFusion->saveDepth();
    if (pangolin::Pushed(*gui->savePoses)) coFusion->exportPoses();
    if (pangolin::Pushed(*gui->saveView)) {
      static int index = 0;
      std::string viewPath;
      do {
        viewPath = exportDir + "/view" + std::to_string(index++);
      } while (boost::filesystem::exists(viewPath + ".png"));
      gui->saveColorImage(viewPath);
    }

    TOCK("GUI");
  }
  if (exportPoses) coFusion->exportPoses();
  if (exportModels) coFusion->savePly();
}

void MainController::drawScene(DRAW_COLOR_TYPE backgroundColor, DRAW_COLOR_TYPE objectColor) {
  if (gui->followPose->Get()) {
    pangolin::OpenGlMatrix mv;

    Eigen::Matrix4f currPose = coFusion->getCurrPose();
    Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

    Eigen::Quaternionf currQuat(currRot);
    Eigen::Vector3f forwardVector(0, 0, 1);
    Eigen::Vector3f upVector(0, iclnuim ? 1 : -1, 0);

    Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
    Eigen::Vector3f up = (currQuat * upVector).normalized();

    Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

    eye -= forward;

    Eigen::Vector3f at = eye + forward;

    Eigen::Vector3f z = (eye - at).normalized();   // Forward
    Eigen::Vector3f x = up.cross(z).normalized();  // Right
    Eigen::Vector3f y = z.cross(x);

    Eigen::Matrix4d m;
    m << x(0), x(1), x(2), -(x.dot(eye)), y(0), y(1), y(2), -(y.dot(eye)), z(0), z(1), z(2), -(z.dot(eye)), 0, 0, 0, 1;

    memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

    gui->s_cam.SetModelViewMatrix(mv);
  }

  gui->preCall();

  Eigen::Matrix4f pose = coFusion->getCurrPose();
  Eigen::Matrix4f viewprojection =
      Eigen::Map<Eigen::Matrix<pangolin::GLprecision, 4, 4>>(gui->s_cam.GetProjectionModelViewMatrix().m).cast<float>();

  if (gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get()) {
    coFusion->computeFeedbackBuffers();
  }

  if (gui->drawRawCloud->Get()) {
    coFusion->getFeedbackBuffers()
        .at(FeedbackBuffer::RAW)
        ->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
  }

  if (gui->drawFilteredCloud->Get()) {
    coFusion->getFeedbackBuffers()
        .at(FeedbackBuffer::FILTERED)
        ->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
  }

  if (false) {
    glFinish();
    TICK("FXAA");

    gui->drawFXAA(viewprojection,  // gui->s_cam.GetProjectionModelViewMatrix(),
                  pose, gui->s_cam.GetModelViewMatrix(), coFusion->getModels(), coFusion->getTick(), coFusion->getTimeDelta(), iclnuim);

    TOCK("FXAA");

    glFinish();
  } else {
    int selectedColorType =
        gui->drawNormals->Get() ? 1 : gui->drawColors->Get() ? 2 : gui->drawTimes->Get() ? 3 : gui->drawLabelColors->Get() ? 4 : 2;
    int globalColorType = selectedColorType;
    int objectColorType = selectedColorType;
    if (backgroundColor != DRAW_USER_DEFINED) globalColorType = backgroundColor;
    if (objectColor != DRAW_USER_DEFINED) objectColorType = objectColor;

    if (gui->drawGlobalModel->Get()) {
      coFusion->getBackgroundModel()->renderPointCloud(viewprojection, gui->drawUnstable->Get(), gui->drawPoints->Get(),
                                                       gui->drawWindow->Get(), globalColorType, coFusion->getTick(),
                                                       coFusion->getTimeDelta());
    }

    auto itBegin = coFusion->getModels().begin();
    itBegin++;  // Skip global
    auto itEnd = coFusion->getModels().end();
    // int i = 0;
    for (auto model = itBegin; model != itEnd; model++) {
      if (gui->drawObjectModels->Get()) {
        (*model)->renderPointCloud(viewprojection * pose * (*model)->getPose().inverse(), gui->drawUnstable->Get(), gui->drawPoints->Get(),
                                   gui->drawWindow->Get(), objectColorType, coFusion->getTick(), coFusion->getTimeDelta());

        glFinish();
      }
    }
  }
  if (gui->drawPoseLog->Get()) {
    bool object = false;
    for (auto& model : coFusion->getModels()) {
      const std::vector<Model::PoseLogItem>& poseLog = model->getPoseLog();

      glColor3f(0, 1, 1);
      glBegin(GL_LINE_STRIP);
      for (const auto& item : poseLog) {
        glVertex3f(item.p(0), item.p(1), item.p(2));
      }
      glEnd();
      if (object) {
        glColor3f(0, 1, 0.2);
        gui->drawFrustum(pose * model->getPose().inverse());
        glColor3f(1, 1, 0.2);
      }
      object = true;
    }
  }

  const bool drawCamera = true;
  if (drawCamera) {
    coFusion->getLost() ? glColor3f(1, 1, 0) : glColor3f(1, 0, 1);
    gui->drawFrustum(pose);

    // Draw axis
    Eigen::Matrix4f wtoc = pose;
    float vlength = 0.07;
    Eigen::Vector4f c = wtoc * Eigen::Vector4f(0, 0, 0, 1);
    Eigen::Vector4f x = wtoc * Eigen::Vector4f(vlength, 0, 0, 1);
    Eigen::Vector4f y = wtoc * Eigen::Vector4f(0, vlength, 0, 1);
    Eigen::Vector4f z = wtoc * Eigen::Vector4f(0, 0, vlength, 1);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(c(0), c(1), c(2));
    glVertex3f(x(0), x(1), x(2));
    glColor3f(0, 1, 0);
    glVertex3f(c(0), c(1), c(2));
    glVertex3f(y(0), y(1), y(2));
    glColor3f(0, 0, 1);
    glVertex3f(c(0), c(1), c(2));
    glVertex3f(z(0), z(1), z(2));
    glEnd();
  }
  glColor3f(1, 1, 1);

  if (gui->drawFerns->Get()) {
    glColor3f(0, 0, 0);
    for (size_t i = 0; i < coFusion->getFerns().frames.size(); i++) {
      if ((int)i == coFusion->getFerns().lastClosest) continue;

      gui->drawFrustum(coFusion->getFerns().frames.at(i)->pose);
    }
    glColor3f(1, 1, 1);
  }

  if (gui->drawDefGraph->Get()) {
    const std::vector<GraphNode*>& graph = coFusion->getLocalDeformation().getGraph();

    for (size_t i = 0; i < graph.size(); i++) {
      pangolin::glDrawCross(graph.at(i)->position(0), graph.at(i)->position(1), graph.at(i)->position(2), 0.1);

      for (size_t j = 0; j < graph.at(i)->neighbours.size(); j++) {
        pangolin::glDrawLine(graph.at(i)->position(0), graph.at(i)->position(1), graph.at(i)->position(2),
                             graph.at(graph.at(i)->neighbours.at(j))->position(0), graph.at(graph.at(i)->neighbours.at(j))->position(1),
                             graph.at(graph.at(i)->neighbours.at(j))->position(2));
      }
    }
  }

  if (coFusion->getFerns().lastClosest != -1) {
    glColor3f(1, 0, 0);
    gui->drawFrustum(coFusion->getFerns().frames.at(coFusion->getFerns().lastClosest)->pose);
    glColor3f(1, 1, 1);
  }

  const std::vector<PoseMatch>& poseMatches = coFusion->getPoseMatches();

  int maxDiff = 0;
  for (size_t i = 0; i < poseMatches.size(); i++) {
    if (poseMatches.at(i).secondId - poseMatches.at(i).firstId > maxDiff) {
      maxDiff = poseMatches.at(i).secondId - poseMatches.at(i).firstId;
    }
  }

  for (size_t i = 0; i < poseMatches.size(); i++) {
    if (gui->drawDeforms->Get()) {
      if (poseMatches.at(i).fern) {
        glColor3f(1, 0, 0);
      } else {
        glColor3f(0, 1, 0);
      }
      for (size_t j = 0; j < poseMatches.at(i).constraints.size(); j++) {
        pangolin::glDrawLine(poseMatches.at(i).constraints.at(j).sourcePoint(0), poseMatches.at(i).constraints.at(j).sourcePoint(1),
                             poseMatches.at(i).constraints.at(j).sourcePoint(2), poseMatches.at(i).constraints.at(j).targetPoint(0),
                             poseMatches.at(i).constraints.at(j).targetPoint(1), poseMatches.at(i).constraints.at(j).targetPoint(2));
      }
    }
  }
  glColor3f(1, 1, 1);

  if (!showcaseMode) {
    // Generate textures, which are specifically for visualisation
    coFusion->normaliseDepth(0.3f, gui->depthCutoff->Get());
    coFusion->coloriseMasks();

    // Render textures to viewports
    for (std::map<std::string, GPUTexture*>::const_iterator it = coFusion->getTextures().begin(); it != coFusion->getTextures().end();
         ++it) {
      if (it->second->draw) {
        gui->displayImg(it->first, it->second);
      }
    }

    // gui->displayImg("IcpError", eFusion->getTextures()["ICP_ERROR"]);
    gui->displayImg("ModelImg", coFusion->getIndexMap().getSplatImageTex());
    // gui->displayImg("Model", eFusion->getIndexMap().getDrawTex());

    auto itBegin = coFusion->getModels().begin();
    auto itEnd = coFusion->getModels().end();
    int i = 0;
    for (auto model = itBegin; model != itEnd; model++) {
      gui->displayImg("ICP" + std::to_string(++i), (*model)->getICPErrorTexture());
      // gui->displayImg("P" + std::to_string(i), (*model)->getUnaryConfTexture());
      if (i >= 4) break;
    }
    for (; i < 4;) {
      gui->displayEmpty("ICP" + std::to_string(++i));
      gui->displayEmpty("P" + std::to_string(i));
    }
  }

  std::stringstream strs;
  strs << coFusion->getBackgroundModel()->lastCount();

  gui->totalPoints->operator=(strs.str());

  std::stringstream strs2;
  strs2 << coFusion->getLocalDeformation().getGraph().size();

  gui->totalNodes->operator=(strs2.str());

  std::stringstream strs3;
  strs3 << coFusion->getFerns().frames.size();

  gui->totalFerns->operator=(strs3.str());

  std::stringstream strs4;
  strs4 << coFusion->getDeforms();

  gui->totalDefs->operator=(strs4.str());

  std::stringstream strs5;
  strs5 << coFusion->getTick() << "/" << logReader->getNumFrames();

  gui->logProgress->operator=(strs5.str());

  std::stringstream strs6;
  strs6 << coFusion->getFernDeforms();

  gui->totalFernDefs->operator=(strs6.str());

  gui->postCall();
}
