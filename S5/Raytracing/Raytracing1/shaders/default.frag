#version 430

layout (location = 0) uniform int nbSpheres;
layout (location = 1) uniform int nbPlanes;
layout (location = 2) uniform int nbMeshTriangles;
layout (location = 3) uniform float view[16];
layout (location = 19) uniform vec4 spheres[256];
layout (location = 276) uniform vec3 planes[256];
layout (location = 534) uniform vec3 meshes[256];

layout (location = 800) uniform vec4 light;
layout (location = 801) uniform vec3 light_color;
layout (location = 802) uniform vec3 ambient_light;
layout (location = 803) uniform int lighting_type;
layout (location = 804) uniform int sample_rate;
layout (location = 805) uniform vec4 roth_spheres[4];

// Material properties (could be extended to have different materials per object)
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Hit {
    float distance;
    vec3 surface_normal;
    int surface_material_index;
};

struct Roth {
    int nb_hits;
    Hit hits[8];// Max 8 hits points
};

vec2 uv;// the UV coordinates of this pixel on the canvas

out vec4 fColor;// final color

void compute_primary_ray(in vec2 uv, out vec3 ray_pos, out vec3 ray_dir);

vec2 get_uv_plane_size();

float ray_sphere(vec3 ray_pos, vec3 ray_dir, vec3 sphere_pos, float sphere_radius, out vec3 intersect_pt, out vec3 normal);// Test ray-sphere intersection, if intersect: return distance, point and normal

float ray_triangle(vec3 ray_pos, vec3 ray_dir, vec3 p0, vec3 p1, vec3 p2, out vec3 intersect_pt, out vec3 normal);// test ray-triangle intersection, if intersect: return distance, point and normal

float ray_plane(vec3 ray_pos, vec3 ray_dir, vec3 plane_pos, vec3 plane_normal, out vec3 intersec_pt, out vec3 normal);// test ray–plane intersection , if intersect : return distance, point and normal

float compute_nearest_intersection(vec3 ray_pos, vec3 ray_dir, out vec3 intersec_i, out vec3 normal_i, out int object_id, out int object_type);// find nearest intersection in the scene, , if intersect: return distance, point and normal

vec2 compute_uv();

vec3 raycast(vec2 uv);// compute primary ray, computeIntersection, shade, return color

vec3 calculate_lighting(vec3 position, vec3 normal, vec3 view_dir, Material material);

vec3 phong_brdf(vec3 light_pos, vec3 normal, vec3 view_dir, Material material);

vec3 blinn_brdf(vec3 light_pos, vec3 normal, vec3 view_dir, Material material);

Material get_material(int object_type, int object_id);

float calculate_shadows(vec3 position, vec3 light_dir, float light_distance);

Roth unionCSG(Roth roth1, Roth roth2);
Roth intersectionCSG(Roth roth1, Roth roth2, vec3 rey_dir);
Roth complementCSG(Roth roth1);
Roth differenceCSG(Roth roth1, Roth roth2, vec3 ray_dir);
Roth ray_sphere_roth(vec3 ray_pos, vec3 ray_dir, vec3 sphere_pos, float sphere_radius, int material_index);
float rayCSG(vec3 ray_pos, vec3 ray_dir, out vec3 intersect_point, out vec3 normal, out int object_id, out int object_type);

void main() {
    vec3 color = vec3(0.0);
    int samples = max(1, sample_rate);

    float step_size = 1.0 / samples;

    for (int y = 0; y < samples; y++) {
        for (int x = 0; x < samples; x++) {
            vec2 offset = vec2(
            (float(x) + 0.5) * step_size - 0.5,
            (float(y) + 0.5) * step_size - 0.5
            ) / vec2(view[0], view[1]);

            vec2 sample_coord = gl_FragCoord.xy + offset * vec2(view[0], view[1]);
            vec2 sample_uv = (sample_coord / vec2(view[0], view[1]) - 0.5) * 2.0;

            float aspect = view[0] / view[1];
            if (aspect >= 1.0) {
                sample_uv.x *= aspect;
            } else {
                sample_uv.y /= aspect;
            }

            // Accumulate color from this sample
            color += raycast(sample_uv);
        }
    }

    // Average the accumulated color values
    color /= float(samples * samples);

    // Output final color
    fColor = vec4(color, 1.0f);
    //    
    //    uv = compute_uv();
    //    fColor = vec4(raycast(uv), 1.0f);
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

float ray_triangle(vec3 ray_pos, vec3 ray_dir, vec3 p0, vec3 p1, vec3 p2, out vec3 intersect_pt, out vec3 normal){
    // Calculate triangle normal
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    normal = normalize(cross(edge1, edge2));

    // Check if ray and triangle are parallel
    float ndotray = dot(normal, ray_dir);
    if (abs(ndotray) < 0.000001)
    return -1.0;// They are parallel, no intersection

    // Calculate distance from ray origin to triangle plane
    float d = dot(normal, p0);
    float t = (dot(normal, ray_pos) - d) / -ndotray;

    // Check if triangle is behind the ray
    if (t < 0.0)
    return -1.0;

    // Calculate intersection point
    intersect_pt = ray_pos + t * ray_dir;

    // Check if intersection point is inside the triangle
    // Using barycentric coordinates
    vec3 c0 = cross(p1 - intersect_pt, p2 - intersect_pt);
    vec3 c1 = cross(p2 - intersect_pt, p0 - intersect_pt);
    vec3 c2 = cross(p0 - intersect_pt, p1 - intersect_pt);

    // Check if all vectors point in same direction as normal
    // (intersection point is inside triangle)
    if (dot(normal, c0) < 0.0 || dot(normal, c1) < 0.0 || dot(normal, c2) < 0.0)
    return -1.0;

    return t;
}

float ray_plane(vec3 ray_pos, vec3 ray_dir, vec3 plane_pos, vec3 plane_normal, out vec3 intersec_pt, out vec3 normal){
    // Normalize the plane normal
    plane_normal = normalize(plane_normal);

    // Check if ray and plane are parallel
    float denom = dot(plane_normal, ray_dir);
    if (abs(denom) < 0.000001) {
        return -1.0;// No intersection, ray is parallel to plane
    }

    // Calculate distance to intersection point
    float t = dot(plane_normal, plane_pos - ray_pos) / denom;

    // Check if plane is behind the ray
    if (t < 0.0) {
        return -1.0;
    }

    // Calculate intersection point
    intersec_pt = ray_pos + t * ray_dir;

    // Set output normal (same as plane normal if ray hits front side)
    normal = denom < 0.0 ? plane_normal : -plane_normal;

    return t;
}

float compute_nearest_intersection(vec3 ray_pos, vec3 ray_dir, out vec3 intersec_i, out vec3 normal_i, out int object_id, out int object_type){
    float dist = 1e10f;
    bool hit = false;

    // Test intersection with sphere
    for (int i = 0; i < nbSpheres && i < 256; i++){
        vec3 sphere_pos = spheres[i].xyz;
        float sphere_radius = spheres[i].w;
        vec3 intersec_point_sphere;
        vec3 normal_sphere;
        float sphere_dist = ray_sphere(ray_pos, ray_dir, sphere_pos, sphere_radius, intersec_point_sphere, normal_sphere);
        if (sphere_dist > 0.0 && sphere_dist < dist) {
            intersec_i = intersec_point_sphere;
            normal_i = normal_sphere;
            dist = sphere_dist;
            object_id = i;
            object_type = 0;
            hit = true;
        }
    }


    // Test intersection with a ground plane
    for (int i = 0; i < nbPlanes && i < 128; i++){
        vec3 plane_pos = planes[i * 2];
        vec3 plane_normal = planes[i * 2 + 1];
        vec3 intersec_point_plane;
        vec3 normal_plane;
        float plane_dist = ray_plane(ray_pos, ray_dir, plane_pos, plane_normal, intersec_point_plane, normal_plane);
        if (plane_dist > 0.0 && plane_dist < dist) {
            intersec_i = intersec_point_plane;
            normal_i = normal_plane;
            dist = plane_dist;
            object_id = i;
            object_type = 1;
            hit = true;
        }
    }

    // Test intersection with a triangle
    for (int i = 0; i < nbMeshTriangles && i < 85; i++){
        vec3 p0 = meshes[i * 3];
        vec3 p1 = meshes[i * 3 + 1];
        vec3 p2 = meshes[i * 3 + 2];
        vec3 intersec_point_triangle;
        vec3 normal_triangle;
        float triangle_dist = ray_triangle(ray_pos, ray_dir, p0, p1, p2, intersec_point_triangle, normal_triangle);
        if (triangle_dist > 0.0 && triangle_dist < dist) {
            intersec_i = intersec_point_triangle;
            normal_i = normal_triangle;
            dist = triangle_dist;
            object_id = i;
            object_type = 2;
            hit = true;
        }
    }

    vec3 csg_intersect_point;
    vec3 csg_normal;
    int csg_object_id, csg_object_type;
    float csg_dist = rayCSG(ray_pos, ray_dir, csg_intersect_point, csg_normal, csg_object_id, csg_object_type);

    if ((csg_dist > 0.0) && (dist < 0.0 || csg_dist < dist)){
        intersec_i = csg_intersect_point;
        normal_i = csg_normal;
        object_id = csg_object_id;
        object_type = csg_object_type;
        dist = csg_dist;
    }

    if (!hit) return -1.0;
    return dist;
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

    // Then try regular scene intersection
    vec3 scene_intersect_point;
    vec3 scene_normal;
    int scene_object_id, scene_object_type;
    float scene_dist = compute_nearest_intersection(ray_pos, ray_dir, scene_intersect_point, scene_normal, scene_object_id, scene_object_type);

    if (scene_dist > 0.0) {
        Material material = get_material(scene_object_type, scene_object_id);
        vec3 view_dir = normalize(ray_pos - scene_intersect_point);
        return calculate_lighting(scene_intersect_point, scene_normal, view_dir, material);
    } else {
        // If no intersection, return background color
        return vec3(0.2, 0.3, 0.4);
    }
}

// Determine if a point is in shadow
float calculate_shadows(vec3 position, vec3 light_dir, float light_distance) {
    //    return false;
    vec3 shadow_intersec;
    vec3 shadow_normal;
    int shadow_obj_id, shadow_obj_type;

    // Offset the ray origin slightly to avoid self-intersection
    vec3 offset_pos = position + light_dir * 0.001;

    // Check if there's any intersection between the point and the light
    float shadow_dist = compute_nearest_intersection(offset_pos, light_dir, shadow_intersec, shadow_normal, shadow_obj_id, shadow_obj_type);

    // If there's an intersection and it's closer than the light, the point is in shadow
    return (shadow_dist > 0.0 && shadow_dist < light_distance) ? distance(position, shadow_intersec) / distance(position, light.xyz) : 1.0;
}

vec3 calculate_lighting(vec3 position, vec3 normal, vec3 view_dir, Material material){
    // Ambient
    vec3 ambient = material.ambient * ambient_light;

    vec3 light_pos = light.xyz;
    float light_intensity = light.w;

    vec3 light_dir = normalize(light_pos - position);
    float light_distance = distance(light_pos, position);
    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * material.diffuse;

    // Specular
    vec3 specular = lighting_type == 0 ? phong_brdf(light_dir, normal, view_dir, material): blinn_brdf(light_dir, normal, view_dir, material);

    // Attenuation (light falloff with distance)
    float attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);

    float sl = calculate_shadows(position, light_dir, light_distance);
    vec3 result = ambient + sl * (diffuse + specular * light_intensity);
    result *= light_color;
    return result;
}

vec3 phong_brdf(vec3 light_dir, vec3 normal, vec3 view_dir, Material material){
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specular = spec * material.specular;
    return specular;
}

vec3 blinn_brdf(vec3 light_dir, vec3 normal, vec3 view_dir, Material material){
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(halfway, normal), 0.0), material.shininess);
    vec3 specular = spec * material.specular;
    return specular;
}

Material get_material(int object_type, int object_id) {
    Material mat;

    // Different material properties based on object type
    if (object_type == 0) { // Sphere
        // Vary material based on sphere ID for visual interest
        int material_variant = object_id % 3;

        if (material_variant == 0) {
            // Plastic-like material
            mat.ambient = vec3(0.1, 0.1, 0.1);
            mat.diffuse = vec3(0.8, 0.2, 0.2);// Red
            mat.specular = vec3(1.0, 1.0, 1.0);
            mat.shininess = 32.0;
        } else if (material_variant == 1) {
            // Light
            mat.ambient = ambient_light;
            mat.diffuse = light_color;
            mat.specular = vec3(0.9, 0.9, 0.9);
            mat.shininess = 128.0;
        } else {
            // Glass-like material
            mat.ambient = vec3(0.1, 0.1, 0.1);
            mat.diffuse = vec3(0.2, 0.2, 0.8);// Blue
            mat.specular = vec3(1.0, 1.0, 1.0);
            mat.shininess = 256.0;
        }
    }
    else if (object_type == 1) { // Plane
        // Checkerboard pattern for planes
        mat.ambient = vec3(0.1, 0.1, 0.1);
        mat.diffuse = vec3(0.9, 0.9, 0.9);// White
        mat.specular = vec3(0.2, 0.2, 0.2);
        mat.shininess = 4.0;
    }
    else if (object_type == 2){ // Triangle/Mesh
        mat.ambient = vec3(0.1, 0.1, 0.1);
        mat.diffuse = vec3(0.8, 0.8, 0.2);// Yellow
        mat.specular = vec3(0.5, 0.5, 0.5);
        mat.shininess = 16.0;
    }
    else {
        mat.ambient = vec3(0.1, 0.1, 0.1);
        mat.specular = vec3(1.0, 1.0, 1.0);
        mat.shininess = 32.0;
        switch (object_id) {
            case 0:
            mat.diffuse = vec3(0.8, 0.2, 0.2);// Red
            break;
            case 1:
            mat.diffuse = vec3(0.8, 0.2, 0.2);// Red
            break;
            case 2:
            mat.diffuse = vec3(0.2, 0.2, 0.8);// Blue
            break;
            case 3:
            mat.diffuse = vec3(0.2, 0.8, 0.2);// Green
            break;
        }
    }

    return mat;
}

Roth ray_sphere_roth(vec3 ray_pos, vec3 ray_dir, vec3 sphere_pos, float sphere_radius, int material_index) {
    Roth result;
    result.nb_hits = 0;

    vec3 oc = ray_pos - sphere_pos;

    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;

    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return result;// No intersection
    }

    // Calculate the two intersection distances
    float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    float t2 = (-b + sqrt(discriminant)) / (2.0 * a);

    // Add entry point
    vec3 intersect_pt1 = ray_pos + t1 * ray_dir;
    vec3 normal1 = normalize(intersect_pt1 - sphere_pos);

    result.hits[result.nb_hits].distance = t1;
    result.hits[result.nb_hits].surface_normal = normal1;
    result.hits[result.nb_hits].surface_material_index = material_index;
    result.nb_hits++;

    // Add exit point
    vec3 intersect_pt2 = ray_pos + t2 * ray_dir;
    vec3 normal2 = normalize(intersect_pt2 - sphere_pos);

    result.hits[result.nb_hits].distance = t2;
    result.hits[result.nb_hits].surface_normal = -normal2;// Negative normal for exit point
    result.hits[result.nb_hits].surface_material_index = material_index;
    result.nb_hits++;

    return result;
}

// Union operation: Combines two objects, returns all intersection points sorted by distance
Roth unionCSG(Roth roth1, Roth roth2) {
    Roth result;
    result.nb_hits = 0;

    int i = 0, j = 0;

    // Merge hits from both roths, keeping them ordered by distance
    int inside_count = 0;
    while (i < roth1.nb_hits && j < roth2.nb_hits && result.nb_hits < 8) {
        if (roth1.hits[i].distance < roth2.hits[j].distance) {
            if (i % 2 != 0){
                if (inside_count == 1){
                    result.hits[result.nb_hits] = roth1.hits[i];
                    result.nb_hits++;

                }
                inside_count--;
            }
            else {
                if (inside_count == 0){
                    result.hits[result.nb_hits] = roth1.hits[i];
                    result.nb_hits++;

                }
                inside_count++;
            }
            i++;
        } else {
            if (j % 2 != 0){
                if (inside_count == 1){
                    result.hits[result.nb_hits] = roth2.hits[j];
                    result.nb_hits++;

                }
                inside_count--;
            }
            else {
                if (inside_count == 0){
                    result.hits[result.nb_hits] = roth2.hits[j];
                    result.nb_hits++;

                }
                inside_count++;
            }
            j++;
        }
    }

    // Add remaining hits from roth1
    while (i < roth1.nb_hits && result.nb_hits < 8) {
        result.hits[result.nb_hits] = roth1.hits[i];
        i++;
        result.nb_hits++;
    }

    // Add remaining hits from roth2
    while (j < roth2.nb_hits && result.nb_hits < 8) {
        result.hits[result.nb_hits] = roth2.hits[j];
        j++;
        result.nb_hits++;
    }

    return result;
}

// Intersection operation: Returns intersection points where both objects overlap
Roth intersectionCSG(Roth roth1, Roth roth2, vec3 ray_dir) {
    Roth result;
    result.nb_hits = 0;

    int i = 0, j = 0;

    // Merge hits from both roths, keeping them ordered by distance
    int inside_count = 0;
    while (i < roth1.nb_hits && j < roth2.nb_hits && result.nb_hits < 8) {
        if (roth1.hits[i].distance < roth2.hits[j].distance) {
            if (i % 2 != 0){
                if (inside_count == 2){
                    result.hits[result.nb_hits] = roth1.hits[i];
                    result.nb_hits++;

                }
                inside_count--;
            }
            else {
                if (inside_count == 1){
                    result.hits[result.nb_hits] = roth1.hits[i];
                    result.nb_hits++;

                }
                inside_count++;
            }
            i++;
        } else {
            if (j % 2 != 0){
                if (inside_count == 2){
                    result.hits[result.nb_hits] = roth2.hits[j];
                    result.nb_hits++;

                }
                inside_count--;
            }
            else {
                if (inside_count == 1){
                    result.hits[result.nb_hits] = roth2.hits[j];
                    result.nb_hits++;

                }
                inside_count++;
            }
            j++;
        }
    }

    return result;
}

// Complement operation: Inverts an object, turning inside to outside
Roth complementCSG(Roth roth) {
    Roth result;

    // For empty Roth, complement is a special case
    if (roth.nb_hits == 0) {
        // Create a "universe" hit at infinity
        result.nb_hits = 2;
        result.hits[0].distance = 0.0;
        result.hits[0].surface_normal = vec3(0.0, 0.0, 0.0);
        result.hits[0].surface_material_index = 0;

        result.hits[1].distance = 1.0e30;// Very far away
        result.hits[1].surface_normal = vec3(0.0, 0.0, 0.0);
        result.hits[1].surface_material_index = 0;
        return result;
    }

    result.nb_hits = roth.nb_hits;

    // Swap entry and exit points by reversing the array
    for (int i = 0; i < roth.nb_hits; i++) {
        result.hits[i] = roth.hits[roth.nb_hits - 1 - i];
        // Invert normal for all hit points
        //        result.hits[i].surface_normal = -result.hits[i].surface_normal;
    }

    return result;
}

// Difference operation: Subtracts the second object from the first
Roth differenceCSG(Roth roth1, Roth roth2, vec3 ray_dir) {
    Roth complement = complementCSG(roth2);
    return intersectionCSG(roth1, complement, ray_dir);
}

// Main CSG ray function that performs (Sphere1 ∩ Sphere2) + Sphere3) – Sphere4
float rayCSG(vec3 ray_pos, vec3 ray_dir, out vec3 intersect_point, out vec3 normal, out int object_id, out int object_type) {
    // Get sphere data from uniform array
    vec3 sphere1_pos = roth_spheres[0].xyz;
    float sphere1_radius = roth_spheres[0].w;

    vec3 sphere2_pos = roth_spheres[1].xyz;
    float sphere2_radius = roth_spheres[1].w;

    vec3 sphere3_pos = roth_spheres[2].xyz;
    float sphere3_radius = roth_spheres[2].w;

    vec3 sphere4_pos = roth_spheres[3].xyz;
    float sphere4_radius = roth_spheres[3].w;

    // Calculate CSG operations step by step
    // 1. Get intersections with each sphere
    Roth roth1 = ray_sphere_roth(ray_pos, ray_dir, sphere1_pos, sphere1_radius, 0);
    Roth roth2 = ray_sphere_roth(ray_pos, ray_dir, sphere2_pos, sphere2_radius, 1);
    Roth roth3 = ray_sphere_roth(ray_pos, ray_dir, sphere3_pos, sphere3_radius, 2);
    Roth roth4 = ray_sphere_roth(ray_pos, ray_dir, sphere4_pos, sphere4_radius, 3);

    // 2. Perform (Sphere1 ∩ Sphere2) + Sphere3) – Sphere4
    Roth intersection_result = intersectionCSG(roth1, roth2, ray_dir);// Sphere1 ∩ Sphere2
    Roth union_result = unionCSG(intersection_result, roth3);// (Sphere1 ∩ Sphere2) + Sphere3
    Roth final_result = differenceCSG(union_result, roth4, ray_dir);// ((Sphere1 ∩ Sphere2) + Sphere3) - Sphere4

    // 3. Find closest hit point in the final result
    if (final_result.nb_hits > 0) {
        intersect_point = ray_pos + final_result.hits[0].distance * ray_dir;
        normal = final_result.hits[0].surface_normal;
        object_id = final_result.hits[0].surface_material_index;
        object_type = 3;// Special type for CSG objects
        return final_result.hits[0].distance;
    }

    return -1.0;// No intersection
}