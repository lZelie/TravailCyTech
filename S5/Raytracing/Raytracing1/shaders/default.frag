#version 430

layout (location = 0) uniform float view[16];
layout (location = 16) uniform float sphere[16];

vec2 uv;// the UV coordinates of this pixel on the canvas

out vec4 fColor;// final color

void compute_primary_ray(in vec2 uv, out vec3 ray_pos, out vec3 ray_dir);

vec2 get_uv_plane_size();

float ray_sphere(vec3 ray_pos, vec3 ray_dir, vec3 sphere_pos, float sphere_radius, out vec3 intersect_pt, out vec3 normal);// Test ray-sphere intersection, if intersect: return distance, point and normal

//float compute_intersection(vec3 ray_pos, vec3 ray_dir, out vec3 intersec_i, out vec3 normal_i);// find intersection with the sphere, if intersect: return distance, point and normal

vec2 compute_uv();

vec3 raycast(vec2 uv);// compute primary ray, computeIntersection, shade, return color

void main() {
    uv = compute_uv();
    fColor = vec4(raycast(uv), 1.0f);
}

void compute_primary_ray(in vec2 uv, out vec3 ray_pos, out vec3 ray_dir){
    vec3 from = vec3(view[2], view[3], view[4]);// Camera position
    vec3 to = vec3(view[5], view[6], view[7]);// Camera target
    float fovy = view[8];// Field of view
    vec2 uv_size = get_uv_plane_size();// The size of the image in UV

    // Construct the camera coordinate system
    vec3 forward = normalize(from - to);
    vec3 right = normalize(cross(vec3(0.0f, 1.0f, 0.0f), forward));
    vec3 up = cross(forward, right);

    // Compute the distance ti the image plane
    float dist = uv_size.y / tan(fovy / 2);

    // Construct a primary ray going through UV coordinate
    vec3 direction = uv.x * right + uv.y * up - dist * forward;

    ray_pos = from;
    ray_dir = direction;
}

float ray_sphere(vec3 ray_pos, vec3 ray_dir, vec3 sphere_pos, float sphere_radius, out vec3 intersect_pt, out vec3 normal) {
    // Calculate the vector from ray origin to sphere center
    vec3 oc = ray_pos - sphere_pos;

    // Calculate quadratic equation coefficients
    // For ray-sphere intersection: |ray_pos + t*ray_dir - sphere_pos|^2 = sphere_radius^2
    float a = dot(ray_dir, ray_dir);// Length squared of ray direction
    float b = 2.0 * dot(oc, ray_dir);// 2 * dot product of oc and ray direction
    float c = dot(oc, oc) - sphere_radius * sphere_radius;// Length squared of oc minus radius squared

    // Calculate discriminant
    float discriminant = b * b - 4.0 * a * c;

    // If discriminant is negative, ray doesn't hit the sphere
    if (discriminant < 0.0) {
        return -2.0;// No intersection
    }

    // Calculate the nearest intersection distance
    float t = (-b - sqrt(discriminant)) / (2.0 * a);

    // If t is negative, the sphere is behind the ray
    if (t < 0.0) {
        // Try the other intersection point
        t = (-b + sqrt(discriminant)) / (2.0 * a);
        if (t < 0.0) {
            return -1.0;// Both intersections are behind the ray
        }
    }

    // Calculate the intersection point
    intersect_pt = ray_pos + t * ray_dir;

    // Calculate the normal at the intersection point
    normal = normalize(intersect_pt - sphere_pos);

    // Return the intersection distance
    return t;
}

vec2 get_uv_plane_size() {
    float aspect = view[0] / view[1];// width / height

    vec2 uv_size;
    if (aspect >= 1.0) {
        // Width is larger than height
        uv_size = vec2(2.0 * aspect, 2.0);
    } else {
        // Height is larger than width
        uv_size = vec2(2.0, 2.0 / aspect);
    }

    return uv_size;
}

vec2 compute_uv(){
    // Get the aspect ratio (width / height)
    float aspect = view[0] / view[1];

    // Calculate normalized device coordinates (NDC) in the range [-1, 1]
    vec2 ndc = (gl_FragCoord.xy / vec2(view[0], view[1]) - 0.5) * 2.0;

    // Scale coordinates so the shorter dimension has length 1 and
    // aspect ratio is maintained, ensuring a perfect circle
    if (aspect >= 1.0) {
        // Width is larger, scale X by 1/aspect ratio
        return vec2(ndc.x * aspect, ndc.y);
    } else {
        // Height is larger, scale Y by aspect ratio
        return vec2(ndc.x, ndc.y / aspect);
    }
}

vec3 raycast(vec2 uv){
    vec3 ray_pos;
    vec3 ray_dir;
    compute_primary_ray(uv, ray_pos, ray_dir);

    vec3 intersec_point;
    float dist = 1e10f;
    vec3 normal = vec3(1.0f, 1.0f, 1.0f);
    vec3 sphere_pos = vec3(sphere[0], sphere[1], sphere[2]);
    float sphere_radius = sphere[3];
    vec3 intersec_point_sphere;
    vec3 normal_sphere;
    float sphere_dist = ray_sphere(ray_pos, ray_dir, sphere_pos, sphere_radius, intersec_point_sphere, normal_sphere);
    if (sphere_dist > 0 && sphere_dist < dist) {
        intersec_point = intersec_point_sphere;
        dist = sphere_dist;
        normal = normal_sphere;

        // Remap normal from [-1,1] to [0,1] range for visualization
        normal = 0.5 * (normal + 1.0);
    } else {
        // If no intersection, return background color
        return vec3(0.2, 0.3, 0.4); // Example background color
    }

    return normal;
}