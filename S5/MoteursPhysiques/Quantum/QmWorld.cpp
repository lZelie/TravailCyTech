#include <iostream>

#include "QmWorld.h"

#include <thread>

#include "QmParticle.h"

using namespace Quantum;

QmWorld::QmWorld()
{
    std::cout << "Starting Quantum Physics engine." << std::endl;
}

QmWorld::QmWorld(const bool use_delta, const float delta, const bool useRK4): QmWorld()
{
    this->use_delta = use_delta;
    this->delta = delta;
    this->useRK4 = useRK4;
}

QmWorld::~QmWorld() = default;

void QmWorld::simulate(const float t)
{
    time += t;
    auto dt = time - tick_time;
    if (use_delta)
    {
        while (dt >= delta)
        {
            dt = useRK4 ? tickRK4(delta) : tick(delta);
        }
        interpolate(dt);
    }
    else
    {
        useRK4 ? tickRK4(t) : tick(t);
        interpolate(0);
    }
}


void QmWorld::integrate(const float t)
{
    // time += t;
    for (QmBody* b : bodies)
    {
        b->integrate(t);
    }
}

void QmWorld::integrate(const float t, unsigned int i)
{
    // time += t;
    for (QmBody* b : bodies)
    {
        b->integrate(t, i);
    }
}

void QmWorld::integrateRK4(const float t)
{
    for (QmBody* b : bodies)
    {
        b->integrateRK4(t);
    }
}

void QmWorld::computeAccelerations(const unsigned int i)
{
    for (QmBody* b : bodies)
    {
        b->computeAccelerations(i);
    }
}

std::vector<QmContact> QmWorld::broadPhase()
{
    std::vector<QmContact> contacts;
    for (QmBody* b1 : bodies)
    {
        for (QmBody* b2 : bodies)
        {
            if (b1 != b2 && std::ranges::find_if(contacts, [b1, b2](const QmContact& c)
            {
                return b1 == &c.body1 && b2 == &c.body2 || b2 == &c.body1 && b1 ==
                    &c.body2;
            }) == contacts.end())
            {
                const glm::vec3& center1 = b1->getPosition();
                const glm::vec3& center2 = b2->getPosition();
                if (const float distance = glm::distance(center1, center2); distance <= b1->getRadius() + b2->
                    getRadius())
                {
                    contacts.emplace_back(*b1, *b2);
                }
            }
        }
    }
    return contacts;
}

std::vector<QmContact> QmWorld::narrowPhase(const std::vector<QmContact>& contacts)
{
    std::vector narrow(contacts);
    std::vector<std::thread> threads;
    for (QmContact& contact : narrow)
    {
        threads.emplace_back([&]
        {
            std::unique_lock lock1{contact.body1.mutex};
            std::unique_lock lock2{contact.body2.mutex};
            const auto& center1 = contact.body1.getPosition();
            const auto& center2 = contact.body2.getPosition();
            if (const float distance = glm::distance(center1, center2); distance <= contact.body1.getRadius() + contact.body2.getRadius())
            {
                contact.depth = contact.body1.getRadius() + contact.body2.getRadius() - distance;
                contact.normal = (contact.body1.getPosition() - contact.body2.getPosition()) / distance;
            }
        });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    return narrow;
}

void QmWorld::resolve(const std::vector<QmContact>& contacts)
{
    std::vector<std::thread> threads;
    for (const QmContact& contact : contacts)
    {
        threads.emplace_back([&]
        {
            {
                const std::unique_lock lockA{contact.body1.mutex};
                const std::unique_lock lockB{contact.body2.mutex};

                const auto ma = contact.body1.getMass();
                const auto mb = contact.body2.getMass();
                const auto va1 = dot(contact.body1.getVelocity(), contact.normal);
                const auto vb1 = dot(contact.body2.getVelocity(), contact.normal);
                // Calculate impulse
                const auto impulse = -((ma * mb) / (ma + mb)) * (va1 - vb1) * contact.normal;

                contact.body1.setVelocity(contact.body1.getVelocity() + impulse / ma);
                contact.body1.setPosition(
                    contact.body1.getPosition() + contact.depth * (mb / (ma + mb)) * contact.normal);
                contact.body2.setPosition(
                    contact.body2.getPosition() + contact.depth * (ma / (mb + ma)) * -contact.normal);
                contact.body2.setVelocity(contact.body2.getVelocity() - impulse / mb);
            }
        });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
}

void QmWorld::addBody(QmBody* b)
{
    bodies.push_back(b);
}

std::vector<QmBody*> QmWorld::getBodies() const
{
    return bodies;
}

std::vector<QmForceRegistry> QmWorld::getForceRegistry() const
{
    return forceRegistries;
}

std::vector<QmForceRegistry>& QmWorld::getForceRegistry()
{
    return forceRegistries;
}

void QmWorld::setDelta(const float delta)
{
    this->delta = delta;
}

void QmWorld::setCollision(const bool collision)
{
    this->collision = collision;
}

bool QmWorld::isCollision() const
{
    return collision;
}

void QmWorld::clear()
{
    for (const QmBody* b : bodies)
    {
        delete b;
    }
    forceRegistries.clear();
    bodies.clear();
}

void QmWorld::addForceRegistry(const QmForceRegistry& forceRegistry)
{
    forceRegistries.push_back(forceRegistry);
}

void QmWorld::resetBodies()
{
    for (const auto& b : bodies)
    {
        b->reset();
    }
}

float QmWorld::tick(const float t)
{
    resetBodies();
    updateForces(0);
    integrate(t);
    if (collision)
    {
        resolve(narrowPhase(broadPhase()));
    }
    tick_time += t;
    return time - tick_time;
}

float QmWorld::tickRK4(float t)
{
    resetBodies();

    updateForces(0);
    computeAccelerations(0);
    integrate(t / 2, 1);
    updateForces(1);
    computeAccelerations(1);
    integrate(t / 2, 2);
    updateForces(2);
    computeAccelerations(2);
    integrate(t, 3);
    updateForces(3);
    computeAccelerations(3);
    integrateRK4(t);
    if (collision)
    {
        resolve(narrowPhase(broadPhase()));
    }
    tick_time += t;
    return time - tick_time;
}

void QmWorld::interpolate(const float dt)
{
    for (QmBody* b : bodies)
    {
        if (b->getType() == TYPE_PARTICLE)
        {
            const auto* p = reinterpret_cast<QmParticle*>(b);
            p->getUpdater()->update(p->getPos()[0] + dt * p->getVel()[0]);
        }
    }
}

void QmWorld::updateForces(const unsigned int i) const
{
    for (auto& r : forceRegistries)
    {
        r.getForceGenerator()->update(r.getParticle(), i);
    }
}
