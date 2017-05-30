#pragma once

#ifdef NDEBUG
#define checkGLErrors()
#else
#define checkGLErrors() checkGLErrors_(__FILE__, __LINE__)
static void checkGLErrors_(char* file, int line) {
  GLenum err = GL_NO_ERROR;
  while ((err = glGetError()) != GL_NO_ERROR) printf("OpenGL-Error in file %s, line %d: %s\n", file, line, gluErrorString(err));
}
#endif
