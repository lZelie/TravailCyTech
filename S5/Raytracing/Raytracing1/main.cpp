#include "Renderer.h"

int main() {
    Renderer renderer;
    
    // Main render loop
    while (!renderer.shouldClose()) {
        renderer.renderFrame();
    }
    
    return 0;
}