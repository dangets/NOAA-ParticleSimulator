#version 330 compatibility

out vec3 color;

void main() {
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);
    vec3 N;
    N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) {
        discard;
    }
    N.z = sqrt(1.0 - mag);

    float diffuse = max(0.0, dot(lightDir, N));

    color = vec3(1, 0, 0) * diffuse;
}
