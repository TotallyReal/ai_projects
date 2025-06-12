#version 330 core

out vec4 FragColor;
in vec2 fragCoord;

const int MAX_ITERATIONS = 64;
const float RADIUS = 16;
const float SCALE = 5;
const float SMOOTH = 2;

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_dimension;
uniform vec2 u_position;
uniform vec2 u_constant;
uniform float u_color_wave;

/**
 * Compute the number of iterations needed for f^n(x,y) to have norm squared larger than ${RADIUS}.
 * returns (#iteration, first squared norm larger than ${RADIUS}).
 * If this number n is more than ${MAX_ITERATIONS}, return (-1, 1).
 */
vec2 f(float x, float y){
	float xx = 0.0;
	float yy = 0.0;
	for(int k=0;k<MAX_ITERATIONS;k++){
		if (x*x+y*y>float(RADIUS)){
			return vec2(float(k), x*x+y*y);
		}
		xx = x*x-y*y-u_constant[0];
		yy = 2.0*x*y-u_constant[1];
		x = xx;
		y = yy;
	}
	return vec2(-1.,1.);
}

/**
 * Compute the number of iterations needed for f^n(x,y) to have norm squared larger than ${RADIUS}.
 * If this number n is more than ${MAX_ITERATIONS}, return -1.
 */
vec3 f2(float x, float y){
	float xx = 0.0;
	float yy = 0.0;
	for(int k=0;k<MAX_ITERATIONS;k++){
		float d = x*x + y*y;
		if (d>float(RADIUS)){
			return vec3(float(k)/float(MAX_ITERATIONS),float(RADIUS)/d,1.0);
		}
		xx = x*x-y*y-u_constant[0];
		yy = 2.0*x*y-u_constant[1];
		x = xx;
		y = yy;
	}
	return vec3(0,0,0);
}

// convert hsb color vector to rgb.
vec3 hsb2rgb(in vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix( vec3(1.0), rgb, c.y);
}

in vec2 v_uv;
out vec4 fragColor;

void main() {
	vec2 normalizePos = v_uv;
	normalizePos.y = 1.0-normalizePos.y;
	normalizePos = u_dimension*normalizePos+u_position;

	vec2 result = f(normalizePos.x, normalizePos.y);
	float iterations = result.x;
	float norm = result.y;
	if (iterations<0.0){
		//didn't get outside of ${RADIUS} - use black.
		fragColor = vec4(0.0,0.0,0.0,1.0);
		return;
	}
	float inc = min(norm/float(RADIUS)-1.,float(SMOOTH))/float(SMOOTH);
	iterations = iterations - inc;

    vec3 color = vec3(1.-iterations/float(MAX_ITERATIONS));
    if (u_color_wave >= 0){
        color = hsb2rgb(vec3(iterations/float(MAX_ITERATIONS),(1.0-u_color_wave) + u_color_wave*float(RADIUS)/norm,1.0));
    }

	fragColor = vec4(color, 1.0);

}