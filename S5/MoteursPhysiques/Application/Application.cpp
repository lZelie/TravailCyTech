// Physics-engine.cpp?: d?finit le point d'entr?e pour l'application console.
//

#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <glm/glm.hpp>
#include <ctime>
#include <string>

#include "Quantum/Quantum.h"
#include "GxWorld.h"
#include "GxParticle.h"
#include "GxUpdater.h"
#include "Quantum/QmDrag.h"
#include "Quantum/QmGravity.h"
#include "Quantum/QmMagnetism.h"
#include "Quantum/QmFixedMagnetism.h"
#include "Quantum/QmFixedSpring.h"
#include "Quantum/QmSpring.h"

using namespace std;
using namespace Quantum;


GxWorld gxWorld;
QmWorld pxWorld{true, 0.02f, false};

glm::vec3* mousePointer;
int mouseCharge = 0;

int scene = 0;

// ********************** GLUT
// Variables globales

int SCREEN_X = 1024;
int SCREEN_Y = 1024;
int VIEWPORT_X = 5;
int VIEWPORT_Y = 5;
int VIEWPORT_Z = 200;

GLuint DrawListSphere;

int buttons = 0;
float camRotate = 90.f;
float camDist = 30.f;
float camHeight = 0.f;
float mx = 0.f;
float my = 0.f;

// Variables du calcul de framerate
int timeold = 0;
int frame = 0;
float accTime = 0.0;
bool paused = false;

GLfloat light_pos[] = {0.0, 6.0, 1.0, 1.0};

bool gravity = true;
bool drag = true;
bool magnetism = false;
bool spring = false;
bool fixed_position = true;
float damping = 0.999f;
float spring_constant = 10000.f;
std::vector<QmParticle*> other{};

// ********************************************


glm::vec3 randomVector(float min, float max)
{
    float x = min + (max - min) * ((rand() % 100) / 100.f);
    float y = min + (max - min) * ((rand() % 100) / 100.f);
    float z = min + (max - min) * ((rand() % 100) / 100.f);
    return glm::vec3(x, y, z);
}

glm::vec3 randomUpVector(float min, float max)
{
    float x = min + (max - min) * ((rand() % 100) / 100.f) - (max - min) / 2;
    float y = min + (max - min) * ((rand() % 100) / 100.f);
    float z = min + (max - min) * ((rand() % 100) / 100.f) - (max - min) / 2;
    return glm::vec3(x, y, z);
}

QmParticle*
createParticle(glm::vec3 color, glm::vec3 pos, glm::vec3 initSpeed, int charge, std::vector<QmParticle*>& others)
{
    const auto rad = magnetism ? 0.2f : 0.1f + 0.2f * ((rand() % 100) / 100.f);

    GxParticle* g = new GxParticle(color, rad, pos);
    QmParticle* p = new QmParticle(pos, initSpeed, randomVector(0, 0), rad, charge, rad, damping);
    p->setUpdater(new GxUpdater(g));
    gxWorld.addParticle(g);
    pxWorld.addBody(p);
    if (drag) { pxWorld.addForceRegistry(QmForceRegistry(p, new QmDrag(rad, rad))); }
    if (gravity) { pxWorld.addForceRegistry(QmForceRegistry(p, new QmGravity())); }
    if (magnetism)
    {
        if (fixed_position && mousePointer)
        {
            pxWorld.addForceRegistry(QmForceRegistry(p, new QmFixedMagnetism(mousePointer, &mouseCharge, 10000, 100)));
            pxWorld.addForceRegistry(QmForceRegistry(p, new QmMagnetism(others, 10000, 100)));
        }
        else { pxWorld.addForceRegistry(QmForceRegistry(p, new QmMagnetism(others, 10000, 100))); }
    }
    return p;
}

void initScene1()
{
    srand(0);
    printf("Scene 1: Random particles.\n");
    printf("Type space to pause.\n");
    gravity = true;
    magnetism = false;
    spring = false;
    other.clear();
    mousePointer = new glm::vec3(0, 4.5, 0);
}

void initScene2()
{
    mousePointer = new glm::vec3(0, 4.5, 0);
    srand(0);
    printf("Scene 2.\n");
    printf("Magnets.\n");
    gravity = false;
    magnetism = true;
    spring = false;
    other.clear();
    for (int i = 0; i < 100; i++)
    {
        int charge = rand() % 2 * 2 - 1;
        glm::vec3 color{charge == -1 ? 255 : 0, 0.0, charge == 1 ? 255 : 0};
        other.push_back(createParticle(color, randomVector(-10, 10), glm::vec3(0), charge, other));
    }
}

void initScene3()
{
    srand(0);
    printf("Scene 2.\n");
    printf("Empty.\n");
    gravity = true;
    magnetism = false;
    drag = false;
    spring = true;
    other.clear();
    mousePointer = new glm::vec3(0, 4.5, 0);
    float r1 = 0.1f + 0.2f * ((rand() % 100) / 100.f);
    GxParticle* g1 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p1 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p1->setDamping(damping);
    p1->setUpdater(new GxUpdater(g1));
    gxWorld.addParticle(g1);
    pxWorld.addBody(p1);
    pxWorld.addForceRegistry(QmForceRegistry(p1, new QmFixedSpring(*mousePointer, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p1, new QmGravity()));
    GxParticle* g2 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p2 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p2->setDamping(damping);
    p2->setUpdater(new GxUpdater(g2));
    gxWorld.addParticle(g2);
    pxWorld.addBody(p2);
    pxWorld.addForceRegistry(QmForceRegistry(p2, new QmFixedSpring(*mousePointer, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p2, new QmGravity()));
    GxParticle* g3 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p3 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p3->setDamping(damping);
    p3->setUpdater(new GxUpdater(g3));
    gxWorld.addParticle(g3);
    pxWorld.addBody(p3);
    pxWorld.addForceRegistry(QmForceRegistry(p3, new QmFixedSpring(*mousePointer, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p3, new QmGravity()));

    pxWorld.addForceRegistry(QmForceRegistry(p1, new QmSpring(p2, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p2, new QmSpring(p3, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p3, new QmSpring(p1, 1.0f, spring_constant)));

    GxParticle* g4 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p4 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p4->setDamping(damping);
    p4->setUpdater(new GxUpdater(g4));
    gxWorld.addParticle(g4);
    pxWorld.addBody(p4);
    pxWorld.addForceRegistry(QmForceRegistry(p4, new QmSpring(p1, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p4, new QmSpring(p2, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p4, new QmSpring(p3, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p4, new QmGravity()));

    GxParticle* g5 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p5 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p5->setDamping(damping);
    p5->setUpdater(new GxUpdater(g5));
    gxWorld.addParticle(g5);
    pxWorld.addBody(p5);
    pxWorld.addForceRegistry(QmForceRegistry(p5, new QmSpring(p4, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p5, new QmGravity()));

    GxParticle* g6 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p6 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p6->setDamping(damping);
    p6->setUpdater(new GxUpdater(g6));
    gxWorld.addParticle(g6);
    pxWorld.addBody(p6);
    pxWorld.addForceRegistry(QmForceRegistry(p6, new QmSpring(p5, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p6, new QmGravity()));
    GxParticle* g7 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p7 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p7->setDamping(damping);
    p7->setUpdater(new GxUpdater(g7));
    gxWorld.addParticle(g7);
    pxWorld.addBody(p7);
    pxWorld.addForceRegistry(QmForceRegistry(p7, new QmSpring(p5, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p7, new QmGravity()));
    GxParticle* g8 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p8 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p8->setDamping(damping);
    p8->setUpdater(new GxUpdater(g8));
    gxWorld.addParticle(g8);
    pxWorld.addBody(p8);
    pxWorld.addForceRegistry(QmForceRegistry(p8, new QmSpring(p5, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p8, new QmGravity()));

    pxWorld.addForceRegistry(QmForceRegistry(p6, new QmSpring(p7, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p7, new QmSpring(p8, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p8, new QmSpring(p6, 1.0f, spring_constant)));

    GxParticle* g9 = new GxParticle(randomVector(0.1, 1.0), r1, randomVector(0.1, 1.0));
    QmParticle* p9 = new QmParticle(randomVector(0.1, 1.0), randomVector(0, 0), randomVector(0, 0), r1, 0, r1);
    p9->setDamping(damping);
    p9->setUpdater(new GxUpdater(g9));
    gxWorld.addParticle(g9);
    pxWorld.addBody(p9);
    pxWorld.addForceRegistry(QmForceRegistry(p9, new QmSpring(p8, 1.0f, spring_constant)));
    pxWorld.addForceRegistry(QmForceRegistry(p9, new QmGravity()));
}

// ***************************** GLUT methods

void resetView()
{
    camRotate = 90.f;
    camDist = 30.f;
    camHeight = 0.f;
    mx = 0.f;
    my = 0.f;
    if (mousePointer)
    {
        delete mousePointer;
        mousePointer = nullptr;
    }
}

void createDisplayListSphere()
{
    DrawListSphere = glGenLists(1);
    glNewList(DrawListSphere, GL_COMPILE);
    GLUquadricObj* sphere = gluNewQuadric();
    gluQuadricDrawStyle(sphere, GLU_FILL);
    gluQuadricNormals(sphere, GLU_SMOOTH);
    gluSphere(sphere, 1, 12, 8);
    glEndList();
}

void initLight()
{
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHT0);

    GLfloat ambient_light[] = {0.5f, 0.5f, 0.5f, 1.0f};
    GLfloat diffuse_light[] = {0.6f, 0.6f, 0.6f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light);

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
}

void calculateFPS(float dt)
{
    accTime += dt;

    if (accTime > 0.5)
    {
        char t[200];
        sprintf(t, "Quantum Physics Engine: Scene %d, %.2f FPS", scene, frame / accTime);
        glutSetWindowTitle(t);
        accTime = 0.0;
        frame = 0;
    }
}

void idleFunc()
{
    int timer = glutGet(GLUT_ELAPSED_TIME);
    float dt = (float)(timer - timeold) / 1000.f;

    calculateFPS(dt);
    if (!paused) pxWorld.simulate(dt);

    if (mousePointer != nullptr && !magnetism && !spring)
    {
        for (int i = 0; i < 1; i++)
        {
            createParticle(randomVector(1, 0), *mousePointer, randomUpVector(2, 10), rand() % 2 * 2 - 1, other);
        }
    }

    glutPostRedisplay();
    timeold = timer;

}

void drawFunc()
{
    frame++;
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camDist * cos(camRotate), camHeight, camDist * sin(camRotate), 0, camHeight, 0, 0, 1, 0);
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    for (GxParticle* p : gxWorld.getParticles())
    {
        glPushMatrix();
        glm::vec3 color = p->getColor();
        glColor3f(color.x, color.y, color.z);
        glTranslatef(p->getPos().x, p->getPos().y, p->getPos().z);
        glScaled(p->getRadius(), p->getRadius(), p->getRadius());
        glCallList(DrawListSphere);
        glPopMatrix();
    }
    if (mousePointer)
    {
        glPushMatrix();
        glColor3f(1.f, 1.f, 1.f);
        glTranslatef(mousePointer->x, mousePointer->y, mousePointer->z);
        glScaled(0.1f, 0.1f, 0.1f);
        glCallList(DrawListSphere);
        glPopMatrix();
    }
    for (const auto& g : pxWorld.getForceRegistry())
    {
        const auto p1 = g.getParticle()->getPos()[0];
        glm::vec3 p2(0);
        if (g.getForceGenerator()->TYPE == SPRING)
        {
            p2 = reinterpret_cast<QmSpring*>(g.getForceGenerator())->getOther()->getPos()[0];
        }
        else if (g.getForceGenerator()->TYPE == FIXED_SPRING)
        {
            p2 = reinterpret_cast<QmFixedSpring*>(g.getForceGenerator())->getPosition();
        }
        if (length(p2) != 0)
        {
            glBegin(GL_LINES);
            glColor3f(1.f, 1.f, 1.f);
            glVertex3f(p1.x, p1.y, p1.z);
            glVertex3f(p2.x, p2.y, p2.z);
            glEnd();
        }
    }
    glutSwapBuffers();
}


void mouseFunc(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) buttons = 1;
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) buttons = 2;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) buttons = 3;
    if (button == 3) camDist /= 1.07f;
    if (button == 4) camDist *= 1.07f;
    if (state == GLUT_UP) buttons = 0;

    mx = (float)x;
    my = (float)y;
}


void motionFunc(int x, int y)
{
    if (buttons == 1)
    {
        camRotate += (x - mx) / 100.f;
        camHeight += (y - my) * camDist / 1000.f;
    }
    if (buttons == 3)
    {
        if (mousePointer)
            *mousePointer += glm::vec3(x - mx, my - y, 0.f) / 15.f;
    }

    mx = (float)x;
    my = (float)y;
}

void clearWorld()
{
    gxWorld.clear();
    pxWorld.clear();
}

void toggleScene(int s)
{
    clearWorld();
    scene = s;
    resetView();

    switch (scene)
    {
    case 1:
        initScene1();
        break;
    case 2:
        initScene2();
        break;
    case 3:
        initScene3();
        break;
    default:
        break;
    }
}

void update_damping()
{
    std::cout << "Damping = " << damping << std::endl;
    for (const auto& b : pxWorld.getBodies())
    {
        if (b->getType() == TYPE_PARTICLE)
        {
            const auto p = reinterpret_cast<QmParticle*>(b);
            p->setDamping(damping);
        }
    }
}

void update_spring_constant()
{
    std::cout << "Spring constant = " << spring_constant << std::endl;
    for (auto& f : pxWorld.getForceRegistry())
    {
        auto gen = f.getForceGenerator();
        if (gen->TYPE == SPRING)
        {
            reinterpret_cast<QmSpring*>(gen)->setSpringConstant(spring_constant);
        }
        else if (gen->TYPE == FIXED_SPRING)
        {
            reinterpret_cast<QmFixedSpring*>(gen)->setSpringConstant(spring_constant);
        }
    }
}

void remove_gravity()
{
    pxWorld.getForceRegistry().erase(
        std::remove_if(pxWorld.getForceRegistry().begin(), pxWorld.getForceRegistry().end(), [](QmForceRegistry& f)
        {
            return f.getForceGenerator()->TYPE == GRAVITY;
        }), pxWorld.getForceRegistry().end()
    );
}

void add_gravity()
{
    for (const auto& b : pxWorld.getBodies())
    {
        if (b->getType() == TYPE_PARTICLE)
        {
            pxWorld.addForceRegistry(QmForceRegistry(reinterpret_cast<QmParticle*>(b), new QmGravity()));
        }
    }
}

void remove_drag()
{
    pxWorld.getForceRegistry().erase(
        std::remove_if(pxWorld.getForceRegistry().begin(), pxWorld.getForceRegistry().end(), [](QmForceRegistry& f)
        {
            return f.getForceGenerator()->TYPE == DRAG;
        }), pxWorld.getForceRegistry().end()
    );
}

void add_drag()
{
    for (const auto& b : pxWorld.getBodies())
    {
        if (b->getType() == TYPE_PARTICLE)
        {
            auto p = reinterpret_cast<QmParticle*>(b);
            pxWorld.addForceRegistry(QmForceRegistry(p, new QmDrag(p->getMass(), p->getMass())));
        }
    }
}

void keyFunc(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'q':
    case 'Q':
    case 27:
        clearWorld();
        glDeleteLists(DrawListSphere, 1);
        exit(0);
    case '1':
        toggleScene(1);
        break;
    case '2':
        pxWorld.setDelta(0.05f);
        toggleScene(2);
        break;
    case '3':
        pxWorld.setDelta(0.0002f);
        toggleScene(3);
        break;
    case 'g':
        gravity = !gravity;
        if (!gravity)
        {
            remove_gravity();
        }
        else
        {
            add_gravity();
        }
        break;
    case 'd':
        drag = !drag;
        if (!drag)
        {
            remove_drag();
        }
        else
        {
            add_drag();
        }
        break;
    case 'f':
        mouseCharge = mouseCharge == 0 ? 1 : 0;
        break;
    case 'c':
        mouseCharge *= -1;
        break;
    case ' ':
        paused = !paused;
        break;
    case '+':
        damping += 0.001f;
        update_damping();
        break;
    case '-':
        damping -= 0.001f;
        update_damping();
        break;
    case '*':
        spring_constant *= 1.1;
        update_spring_constant();
        break;
    case '/':
        spring_constant *= 0.9;
        update_spring_constant();
        break;
    case 'r':
    case 'R':
        pxWorld.set_use_rk4(!pxWorld.use_rk4());
        break;
    case 'v':
    case 'V':
        pxWorld.setCollision(!pxWorld.isCollision());
    default:
        break;
    }
}

void initGraphics(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(SCREEN_X, SCREEN_Y);
    glutCreateWindow("Quantum Demo");

    glutIdleFunc(idleFunc);
    glutDisplayFunc(drawFunc);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    glutKeyboardFunc(keyFunc);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    //glEnable(GL_PROGRAM_POINT_SIZE);
    //glPointSize(5);
    glMatrixMode(GL_PROJECTION);
    glEnable(GL_DEPTH_TEST);
    glLoadIdentity();
    float aspect = SCREEN_X / (float)SCREEN_Y;
    gluPerspective(45.0, aspect, 1.0, VIEWPORT_Z);
    glViewport(0, 0, SCREEN_X, SCREEN_Y);
    createDisplayListSphere();
    initLight();
}

// ************************** end GLUT methods

int main(int argc, char** argv)
{
    srand(0);
    initGraphics(argc, argv);

    toggleScene(1);

    glutMainLoop();
    return 0;
}
