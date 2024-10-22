#include <iostream>
#include <utility>

#include "GxWorld.h"
#include "GxParticle.h"

GxWorld::GxWorld()
{
	std::cout << "Starting GLUT Graphics engine." << std::endl;
}

GxWorld::~GxWorld()
= default;

void GxWorld::addParticle(GxParticle* p)
{
	particles.push_back(p);
}

std::list<GxParticle*> GxWorld::getParticles()
{
	return particles;
}

void GxWorld::setParticles(std::list<GxParticle*> particles)
{
	this->particles = std::move(particles);
}

void GxWorld::addPlane(const GxPlane& plane)
{
	planes.push_back(plane);
}

std::list<GxPlane> GxWorld::get_planes() const
{
	return planes;
}

void GxWorld::set_planes(const std::list<GxPlane>& planes)
{
	this->planes = planes;
}

void GxWorld::clear()
{
	for (const GxParticle* p : particles)
	{
		delete p;
	}
	particles.clear();
	planes.clear();
}
