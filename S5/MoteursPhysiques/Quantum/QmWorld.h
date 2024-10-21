#ifndef QMWORLD_H
#define QMWORLD_H

#include <list>
#include <vector>
#include <glm/vec3.hpp>

#include "QmContact.h"
#include "QmForceRegistry.h"

namespace Quantum {

	class QmBody;

	class QmWorld {
	public:
		QmWorld();
		QmWorld(bool use_delta, float delta, bool useRK4 = false);
		~QmWorld();
		void simulate(float t);
		void addBody(QmBody*);
        void addForceRegistry(const QmForceRegistry &forceRegistry);
		[[nodiscard]] std::vector<QmBody*> getBodies() const;
		[[nodiscard]] std::vector<QmForceRegistry> getForceRegistry() const;
		[[nodiscard]] std::vector<QmForceRegistry> &getForceRegistry();
		void setDelta(float delta);
		void setCollision(bool collision);
		void setUseDelta(bool useDelta);
		[[nodiscard]] bool isCollision() const;

		void clear();

	private:
		float time = 0.0f;
		float tick_time = 0.0f;
		bool use_delta = false;
		float delta = 0.0f;
		bool useRK4 = false;
		bool collision = true;
		std::vector<QmBody*> bodies;
        std::vector<QmForceRegistry> forceRegistries;
        const glm::vec3 gravity {0.0, -9.81, 0.0};

	public:
		[[nodiscard]] bool use_rk4() const
		{
			return useRK4;
		}

		void set_use_rk4(bool use_rk4)
		{
			useRK4 = use_rk4;
		}

	private:
		void resetBodies();
		float tick(float t);
		float tickRK4(float t);
		void interpolate(float dt);
        void updateForces(unsigned int i) const;
		void integrate(float);
		void integrate(float, unsigned int i);
		void integrateRK4(float t);
		void computeAccelerations(unsigned int i);
		std::vector<QmContact> broadPhase();
		std::vector<QmContact> narrowPhase(const std::vector<QmContact>& contacts);
		void resolve(const std::vector<QmContact>& contacts);
	};

	inline void QmWorld::setUseDelta(bool useDelta)
	{
		this->use_delta = useDelta;
	}
}

#endif