#ifndef GXWORLD_H
#define GXWORLD_H

#include <list>

#include "GxPlane.h"

class GxParticle;

class GxWorld
{
public:
    GxWorld();
    ~GxWorld();
    void addParticle(GxParticle*);
    std::list<GxParticle*> getParticles();
    void setParticles(std::list<GxParticle*>);

    void addPlane(const GxPlane& plane);
    [[nodiscard]] std::list<GxPlane> get_planes() const;

    void set_planes(const std::list<GxPlane>& planes);

    void clear();

private:
    std::list<GxParticle*> particles;
    std::list<GxPlane> planes;
};

#endif
