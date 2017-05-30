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

#include "Model.h"
#include "ModelMatching.h"

Model::GPUSetup::GPUSetup()
    : initProgram(loadProgramFromFile("init_unstable.vert")),
      drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
      drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface.frag", "draw_global_surface.geom")),
      dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
      updateProgram(loadProgramFromFile("update.vert")),
      unstableProgram(loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
      renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
      updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
      updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
      updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT) {
  frameBuffer.AttachColour(*updateMapVertsConfs.texture);
  frameBuffer.AttachColour(*updateMapColorsTime.texture);
  frameBuffer.AttachColour(*updateMapNormsRadii.texture);
  frameBuffer.AttachDepth(renderBuffer);

  updateProgram->Bind();
  int locUpdate[3] = {
      glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"), glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(updateProgram->programId(), 3, locUpdate, GL_INTERLEAVED_ATTRIBS);
  updateProgram->Unbind();

  dataProgram->Bind();
  int dataUpdate[3] = {
      glGetVaryingLocationNV(dataProgram->programId(), "vPosition0"), glGetVaryingLocationNV(dataProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(dataProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(dataProgram->programId(), 3, dataUpdate, GL_INTERLEAVED_ATTRIBS);
  dataProgram->Unbind();

  unstableProgram->Bind();
  int unstableUpdate[3] = {
      glGetVaryingLocationNV(unstableProgram->programId(), "vPosition0"), glGetVaryingLocationNV(unstableProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(unstableProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(unstableProgram->programId(), 3, unstableUpdate, GL_INTERLEAVED_ATTRIBS);
  unstableProgram->Unbind();

  // eraseProgram->Bind();
  // int eraseUpdate[3] = {
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vPosition0"),
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vColor0"),
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vNormRad0"),
  //};
  // glTransformFeedbackVaryingsNV(eraseProgram->programId(), 3, eraseUpdate, GL_INTERLEAVED_ATTRIBS);
  // eraseProgram->Unbind();

  initProgram->Bind();
  int locInit[3] = {
      glGetVaryingLocationNV(initProgram->programId(), "vPosition0"), glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(initProgram->programId(), 3, locInit, GL_INTERLEAVED_ATTRIBS);
  initProgram->Unbind();

  depth_tmp.resize(RGBDOdometry::NUM_PYRS);
  mask_tmp.resize(RGBDOdometry::NUM_PYRS);

  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
    int pyr_rows = Resolution::getInstance().height() >> i;
    int pyr_cols = Resolution::getInstance().width() >> i;

    depth_tmp[i].create(pyr_rows, pyr_cols);
    mask_tmp[i].create(pyr_rows, pyr_cols);
  }
}

#ifdef COFUSION_NUM_SURFELS
const int Model::TEXTURE_DIMENSION = 32 * (int)(sqrt(COFUSION_NUM_SURFELS) / 32);
#else
const int Model::TEXTURE_DIMENSION = 3072;
#endif

const int Model::MAX_VERTICES = Model::TEXTURE_DIMENSION * Model::TEXTURE_DIMENSION;
const int Model::NODE_TEXTURE_DIMENSION = 16384;
const int Model::MAX_NODES = Model::NODE_TEXTURE_DIMENSION / 16;  // 16 floats per node

const int Model::bufferSize = Model::MAX_VERTICES * Vertex::SIZE;

GPUTexture Model::deformationNodes = GPUTexture(NODE_TEXTURE_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

Model::Model(unsigned char id, float confidenceThresh, bool enableFillIn, bool enableErrorRecording, bool enablePoseLogging,
             MatchingType matchingType, float maxDepthThesh)
    : pose(Eigen::Matrix4f::Identity()),
      lastPose(Eigen::Matrix4f::Identity()),
      confidenceThreshold(confidenceThresh),
      maxDepth(maxDepthThesh),
      target(0),
      renderSource(1),
      count(0),
      id(id),
      icpError(enableErrorRecording
                   ? std::make_unique<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_R32F, GL_RED,
                                                  GL_FLOAT, true, true, cudaGraphicsRegisterFlagsSurfaceLoadStore, "ICP")
                   : nullptr),
      rgbError(
          /*enableErrorRecording ? std::make_unique<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_R32F, GL_RED, GL_FLOAT, true, true, cudaGraphicsRegisterFlagsSurfaceLoadStore, "RGB") :*/ nullptr),  // FIXME
      gpu(Model::GPUSetup::getInstance()),
      frameToModel(Resolution::getInstance().width(), Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
                   Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), id),
      fillIn(enableFillIn ? std::make_unique<FillIn>() : nullptr) {
  switch (matchingType) {
    case MatchingType::Drost:
      // removed
      break;
  }

  if (enablePoseLogging) poseLog.reserve(1000);

  float* vertices = new float[bufferSize];

  memset(&vertices[0], 0, bufferSize);

  glGenTransformFeedbacks(1, &vbos[0].stateObject);
  glGenBuffers(1, &vbos[0].dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[0].dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTransformFeedbacks(1, &vbos[1].stateObject);
  glGenBuffers(1, &vbos[1].dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[1].dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];

  memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

  glGenTransformFeedbacks(1, &newUnstableBuffer.stateObject);
  glGenBuffers(1, &newUnstableBuffer.dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().numPixels() * Vertex::SIZE, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  std::vector<Eigen::Vector2f> uv;

  for (int i = 0; i < Resolution::getInstance().width(); i++)
    for (int j = 0; j < Resolution::getInstance().height(); j++)
      uv.push_back(
          Eigen::Vector2f(((float)i / (float)Resolution::getInstance().width()) + 1.0 / (2 * (float)Resolution::getInstance().width()),
                          ((float)j / (float)Resolution::getInstance().height()) + 1.0 / (2 * (float)Resolution::getInstance().height())));

  uvSize = uv.size();

  glGenBuffers(1, &uvo);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  gpu.initProgram->Bind();
  glGenQueries(1, &countQuery);

  // Empty both transform feedbacks
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);
  glDrawArrays(GL_POINTS, 0, 0);  // RUN GPU-PASS
  glEndTransformFeedback();

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, 0);  // RUN GPU-PASS

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.initProgram->Unbind();

  std::cout << "Created model with max number of vertices: " << Model::MAX_VERTICES << std::endl;
}

Model::~Model() {
  glDeleteBuffers(1, &vbos[0].dataBuffer);
  glDeleteTransformFeedbacks(1, &vbos[0].stateObject);

  glDeleteBuffers(1, &vbos[1].dataBuffer);
  glDeleteTransformFeedbacks(1, &vbos[1].stateObject);

  glDeleteQueries(1, &countQuery);

  glDeleteBuffers(1, &uvo);

  glDeleteTransformFeedbacks(1, &newUnstableBuffer.stateObject);
  glDeleteBuffers(1, &newUnstableBuffer.dataBuffer);
}

void Model::initialise(const FeedbackBuffer& rawFeedback, const FeedbackBuffer& filteredFeedback) {
  gpu.initProgram->Bind();

  glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  // It's ok to use either fid because both raw and filtered have the same amount of vertices
  glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);  // RUN GPU-PASS

  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.initProgram->Unbind();

  glFinish();
}

void Model::renderPointCloud(const Eigen::Matrix4f& mvp, const bool drawUnstable, const bool drawPoints, const bool drawWindow,
                             const int colorType, const int time, const int timeDelta) {
  std::shared_ptr<Shader> program = drawPoints ? gpu.drawProgram : gpu.drawSurfelProgram;

  program->Bind();

  // Eigen::Matrix4f mvp = vp;
  // if(id != 0) mvp =  mvp * pose.inverse();

  program->setUniform(Uniform("MVP", mvp));
  program->setUniform(Uniform("threshold", getConfidenceThreshold()));
  program->setUniform(Uniform("colorType", colorType));
  program->setUniform(Uniform("unstable", drawUnstable));
  program->setUniform(Uniform("drawWindow", drawWindow));
  program->setUniform(Uniform("time", time));
  program->setUniform(Uniform("timeDelta", timeDelta));
  program->setUniform(Uniform("maskID", id));

  // This is for the point shader
  program->setUniform(Uniform("pose", (Eigen::Matrix4f)Eigen::Matrix4f::Identity()));
  // TODO

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  program->Unbind();
}

const OutputBuffer& Model::getModel() { return vbos[target]; }

void Model::generateCUDATextures(GPUTexture* depth, GPUTexture* mask) {
  GPUSetup& gpu = GPUSetup::getInstance();
  std::vector<DeviceArray2D<float>>& depthPyr = gpu.depth_tmp;
  std::vector<DeviceArray2D<unsigned char>>& maskPyr = gpu.mask_tmp;

  cudaCheckError();

  depth->cudaMap();
  cudaArray* depthTexturePtr = depth->getCudaArray();
  cudaMemcpy2DFromArray(depthPyr[0].ptr(0), depthPyr[0].step(), depthTexturePtr, 0, 0, depthPyr[0].colsBytes(), depthPyr[0].rows(),
                        cudaMemcpyDeviceToDevice);
  depth->cudaUnmap();

  mask->cudaMap();
  cudaArray* maskTexturePtr = mask->getCudaArray();
  cudaMemcpy2DFromArray(maskPyr[0].ptr(0), maskPyr[0].step(), maskTexturePtr, 0, 0, maskPyr[0].colsBytes(), maskPyr[0].rows(),
                        cudaMemcpyDeviceToDevice);
  mask->cudaUnmap();

  cudaDeviceSynchronize();
  cudaCheckError();

  for (int i = 1; i < RGBDOdometry::NUM_PYRS; ++i) {
    pyrDownGaussF(depthPyr[i - 1], depthPyr[i]);
    pyrDownUcharGauss(maskPyr[i - 1], maskPyr[i]);  // FIXME Better filter
    // TODO Execute in parralel (two cuda streams)
  }
  cudaDeviceSynchronize();
  cudaCheckError();
}

void Model::initICP(bool doFillIn, bool frameToFrameRGB, float depthCutoff, GPUTexture* rgb) {
  TICK("odomInit - Model: " + std::to_string(id));

  // WARNING initICP* must be called before initRGB*
  if (doFillIn) {
    frameToModel.initICPModel(getFillInVertexTexture(), getFillInNormalTexture(), depthCutoff, getPose());
    frameToModel.initRGBModel(getFillInImageTexture());
  } else {
    frameToModel.initICPModel(getVertexConfProjection(), getNormalProjection(), depthCutoff, getPose());
    frameToModel.initRGBModel(frameToFrameRGB && allowsFillIn() ? getFillInImageTexture() : getRGBProjection());
  }

  // frameToModel.initICP(filteredDepth, depthCutoff, mask);
  frameToModel.initICP(gpu.depth_tmp, gpu.mask_tmp, depthCutoff);
  frameToModel.initRGB(rgb);

  TOCK("odomInit - Model: " + std::to_string(id));
}

void Model::performTracking(bool frameToFrameRGB, bool rgbOnly, float icpWeight, bool pyramid, bool fastOdom, bool so3,
                            float maxDepthProcessed, GPUTexture* rgb, int64_t logTimestamp, bool doFillIn) {
  assert(fillIn || !doFillIn);
  lastPose = pose;

  // TODO Allow fillIn again
  initICP(doFillIn, frameToFrameRGB, maxDepthProcessed, rgb);  // TODO: Don't copy RGB

  TICK("odom - Model: " + std::to_string(id));

  Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);

  getFrameOdometry().getIncrementalTransformation(transObject, rotObject, rgbOnly, icpWeight, pyramid, fastOdom, so3,
                                                  icpError->getCudaSurface(), rgbError->getCudaSurface());

  pose.topRightCorner(3, 1) = transObject;
  pose.topLeftCorner(3, 3) = rotObject;

  TOCK("odom - Model: " + std::to_string(id));
}

float Model::computeFusionWeight(float weightMultiplier) const {
  Eigen::Matrix4f diff = getLastTransform();
  Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
  Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

  float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());

  const float largest = 0.01;
  const float minWeight = 0.5;

  if (weighting > largest) weighting = largest;

  weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

  return weighting;
}

void Model::fuse(const int& time, GPUTexture* rgb, GPUTexture* mask, GPUTexture* depthRaw, GPUTexture* depthFiltered,
                 const float depthCutoff, const float weightMultiplier) {
  TICK("Fuse::Data");
  // This first part does data association and computes the vertex to merge with, storing
  // in an array that sets which vertices to update by index
  gpu.frameBuffer.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, gpu.renderBuffer.width, gpu.renderBuffer.height);

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // PROGRAM1: Data association
  gpu.dataProgram->Bind();
  gpu.dataProgram->setUniform(Uniform("cSampler", 0));
  gpu.dataProgram->setUniform(Uniform("drSampler", 1));
  gpu.dataProgram->setUniform(Uniform("drfSampler", 2));
  gpu.dataProgram->setUniform(Uniform("indexSampler", 3));
  gpu.dataProgram->setUniform(Uniform("vertConfSampler", 4));
  gpu.dataProgram->setUniform(Uniform("colorTimeSampler", 5));
  gpu.dataProgram->setUniform(Uniform("normRadSampler", 6));
  gpu.dataProgram->setUniform(Uniform("maskSampler", 7));
  gpu.dataProgram->setUniform(Uniform("time", (float)time));
  gpu.dataProgram->setUniform(Uniform("weighting", computeFusionWeight(weightMultiplier)));
  gpu.dataProgram->setUniform(Uniform("maskID", id));

  gpu.dataProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                             1.0 / Intrinsics::getInstance().fx(), 1.0 / Intrinsics::getInstance().fy())));
  gpu.dataProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.dataProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
  gpu.dataProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.dataProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  gpu.dataProgram->setUniform(Uniform("pose", pose));
  gpu.dataProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableBuffer.stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableBuffer.dataBuffer);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, mask->texture->tid);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, uvSize);  // RUN GPU-PASS

  glEndTransformFeedback();

  gpu.frameBuffer.Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.dataProgram->Unbind();

  glPopAttrib();

  glFinish();
  TOCK("Fuse::Data");

  TICK("Fuse::Update");
  // Next we update the vertices at the indexes stored in the update textures
  // Using a transform feedback conditional on a texture sample

  // PROGRAM2: Fusion
  gpu.updateProgram->Bind();

  gpu.updateProgram->setUniform(Uniform("vertSamp", 0));
  gpu.updateProgram->setUniform(Uniform("colorSamp", 1));
  gpu.updateProgram->setUniform(Uniform("normSamp", 2));
  gpu.updateProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  gpu.updateProgram->setUniform(Uniform("time", time));

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);  // SELECT INPUT

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  // SEE:
  // http://docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/feedbackparticlessample.htm
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);  // SELECT OUTPUT
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);

  // Enter transform feedback mode
  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapVertsConfs.texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapColorsTime.texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapNormsRadii.texture->tid);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // GPU-PASS (target=input)

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.updateProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Update");
}

void Model::clean(  // FIXME what happens with object models and ferns here?
    const int& time, std::vector<float>& graph, const int timeDelta, const float depthCutoff, const bool isFern, GPUTexture* depthFiltered,
    GPUTexture* mask) {
  assert(graph.size() / 16 < MAX_NODES);

  if (graph.size() > 0) {
    // Can be optimised by only uploading new nodes with offset
    glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, graph.size(), 1, GL_LUMINANCE, GL_FLOAT, graph.data());
  }

  TICK("Fuse::Copy");
  // Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
  gpu.unstableProgram->Bind();
  gpu.unstableProgram->setUniform(Uniform("time", time));
  gpu.unstableProgram->setUniform(Uniform("confThreshold", getConfidenceThreshold()));
  gpu.unstableProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.unstableProgram->setUniform(Uniform("outlierCoeff", (float)gpu.outlierCoefficient));
  gpu.unstableProgram->setUniform(Uniform("indexSampler", 0));
  gpu.unstableProgram->setUniform(Uniform("vertConfSampler", 1));
  gpu.unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
  gpu.unstableProgram->setUniform(Uniform("normRadSampler", 3));
  gpu.unstableProgram->setUniform(Uniform("nodeSampler", 4));
  gpu.unstableProgram->setUniform(Uniform("depthSamplerPrediction", 5));
  gpu.unstableProgram->setUniform(Uniform("depthSamplerInput", 6));
  gpu.unstableProgram->setUniform(Uniform("maskSampler", 7));
  gpu.unstableProgram->setUniform(Uniform("nodes", (float)(graph.size() / 16)));
  gpu.unstableProgram->setUniform(Uniform("nodeCols", (float)NODE_TEXTURE_DIMENSION));
  gpu.unstableProgram->setUniform(Uniform("timeDelta", timeDelta));
  gpu.unstableProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));
  gpu.unstableProgram->setUniform(Uniform("isFern", (int)isFern));
  gpu.unstableProgram->setUniform(Uniform("maskID", id));

  Eigen::Matrix4f t_inv = pose.inverse();
  gpu.unstableProgram->setUniform(Uniform("t_inv", t_inv));

  gpu.unstableProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                                 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy())));
  gpu.unstableProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.unstableProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

#if 1
  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);
#endif

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

#if 1
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getDepthTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, mask->texture->tid);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

#if 0
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);
#endif

  glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, newUnstableBuffer.stateObject);  // RUN GPU-PASS

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.unstableProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Copy");
}

void Model::eraseErrorGeometry(GPUTexture* depthFiltered) {
  TICK("Fuse::Erase");

  // Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
  gpu.eraseProgram->Bind();
  gpu.eraseProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.eraseProgram->setUniform(Uniform("indexSampler", 0));
  gpu.eraseProgram->setUniform(Uniform("vertConfSampler", 1));
  gpu.eraseProgram->setUniform(Uniform("colorTimeSampler", 2));
  gpu.eraseProgram->setUniform(Uniform("normRadSampler", 3));
  gpu.eraseProgram->setUniform(Uniform("icpSampler", 4));
  gpu.eraseProgram->setUniform(Uniform("depthSamplerPrediction", 5));
  gpu.eraseProgram->setUniform(Uniform("depthSamplerInput", 6));
  // gpu.unstableProgram->setUniform(Uniform("maskSampler", 7));
  // gpu.unstableProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));
  // gpu.unstableProgram->setUniform(Uniform("maskID", id));

  Eigen::Matrix4f t_inv = pose.inverse();
  gpu.eraseProgram->setUniform(Uniform("t_inv", t_inv));

  gpu.eraseProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                              Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy())));
  gpu.eraseProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.eraseProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

#if 1
  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);
#endif

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

#if 1
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, icpError->texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getDepthTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  // glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

#if 0
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);
#endif

  // glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);
  //
  // glEnableVertexAttribArray(0);
  // glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
  //
  // glEnableVertexAttribArray(1);
  // glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));
  //
  // glEnableVertexAttribArray(2);
  // glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));
  //
  // glDrawTransformFeedback(GL_POINTS, newUnstableBuffer.stateObject); // RUN GPU-PASS

  // glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  // glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.eraseProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Erase");
}

unsigned int Model::lastCount() { return count; }

Eigen::Vector3f Model::rodrigues2(const Eigen::Matrix3f& matrix) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

  double rx = R(2, 1) - R(1, 2);
  double ry = R(0, 2) - R(2, 0);
  double rz = R(1, 0) - R(0, 1);

  double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
  double c = (R.trace() - 1) * 0.5;
  c = c > 1. ? 1. : c < -1. ? -1. : c;

  double theta = acos(c);

  if (s < 1e-5) {
    double t;

    if (c > 0)
      rx = ry = rz = 0;
    else {
      t = (R(0, 0) + 1) * 0.5;
      rx = sqrt(std::max(t, 0.0));
      t = (R(1, 1) + 1) * 0.5;
      ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
      t = (R(2, 2) + 1) * 0.5;
      rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

      if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry * rz > 0)) rz = -rz;
      theta /= sqrt(rx * rx + ry * ry + rz * rz);
      rx *= theta;
      ry *= theta;
      rz *= theta;
    }
  } else {
    double vth = 1 / (2 * s);
    vth *= theta;
    rx *= vth;
    ry *= vth;
    rz *= vth;
  }
  return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

void Model::buildDescription() {
  if (modelMatcher) modelMatcher->buildModelDescription(this);
}

ModelDetectionResult Model::detectInRegion(const FrameData& frame, const cv::Rect& rect) {
  if (modelMatcher) return modelMatcher->detectInRegion(frame, rect);
  return ModelDetectionResult({Eigen::Matrix4f(), false});
}

Model::SurfelMap Model::downloadMap() {
  SurfelMap result;
  result.numPoints = count;
  result.data = std::make_unique<std::vector<Eigen::Vector4f>>();
  result.data->resize(count * 3);  // The compiler should optimise this to be as fast as memset(&vertices[0], 0, count * Vertex::SIZE)

  glFinish();
  GLuint downloadVbo;

  // EFCHANGE Why was this done?
  // glGetBufferSubData(GL_ARRAY_BUFFER, 0, count * Vertex::SIZE, &(result.data->front()));

  glGenBuffers(1, &downloadVbo);
  glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_COPY_READ_BUFFER, vbos[renderSource].dataBuffer);
  glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * Vertex::SIZE);
  glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count * Vertex::SIZE, &(result.data->front()));

  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  glDeleteBuffers(1, &downloadVbo);

  glFinish();

  return result;
}

void Model::performFillIn(GPUTexture* rawRGB, GPUTexture* rawDepth, bool frameToFrameRGB, bool lost) {
  if (fillIn) {
    TICK("FillIn");
    fillIn->vertex(getVertexConfProjection(), rawDepth, lost);
    fillIn->normal(getNormalProjection(), rawDepth, lost);
    fillIn->image(getRGBProjection(), rawRGB, lost || frameToFrameRGB);
    TOCK("FillIn");
  }
}
