
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    //initialize random generator to produce gaussian points later
    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];
    // Creates a normal (Gaussian) distribution for x,y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    num_particles = 50; //set number of particles
    for(unsigned int i =0; i<num_particles; i++){
        Particle particle;
        particle.id = i+1;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;
        
        weights.push_back(1);
        particles.push_back(particle);
    }
    is_initialized = 1;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    double std_x, std_y, std_theta;
    double f_x, f_y, f_theta; //variables for after-movement
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];
    for(unsigned int i =0; i<num_particles; i++){
        if (yaw_rate == 0){//check if the yaw rate is zero to apply equation accordingly
            f_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
            f_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
            f_theta = particles[i].theta;
        }
        else{
            f_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
            f_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta + yaw_rate*delta_t));
            f_theta = particles[i].theta + yaw_rate*delta_t;
        }
        //use Gaussian distributio to count the noise in movement
        normal_distribution<double> dist_x(f_x, std_x);
        normal_distribution<double> dist_y(f_y, std_y);
        normal_distribution<double> dist_theta(f_theta, std_theta);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//  Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for(unsigned int p=0;p<num_particles;p++){
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        vector<LandmarkObs> trans_observations;
        LandmarkObs obs;
        //transform the observation to world coordinate
        for (unsigned int i=0; i<observations.size();i++){
            LandmarkObs trans_obs;
            obs = observations[i];
            trans_obs.x = particles[p].x + cos(particles[p].theta)*obs.x-sin(particles[p].theta)*obs.y;
            trans_obs.y = particles[p].y + sin(particles[p].theta)*obs.x+cos(particles[p].theta)*obs.y;
            trans_observations.push_back(trans_obs);
        }
        particles[p].weight =1.0;
        //calculate weight according to associations
        for (unsigned int j=0;j<trans_observations.size();j++){
            double closet_dis = sensor_range;
            int association =0;
            for (unsigned int i=0; i<map_landmarks.landmark_list.size();i++)
            {
                double landmark_x = map_landmarks.landmark_list[i].x_f;
                double landmark_y = map_landmarks.landmark_list[i].y_f;
                double distance = dist(landmark_x,landmark_y,trans_observations[j].x,trans_observations[j].y);
                if(distance<closet_dis)
                {
                    association = i;
                    closet_dis = distance;
                }
            }
                if(association!=0)
                {
                    double meas_x = trans_observations[j].x;
                    double meas_y = trans_observations[j].y;
                    double mu_x = map_landmarks.landmark_list[association].x_f;
                    double mu_y = map_landmarks.landmark_list[association].y_f;
                    double gauss_norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
                    double exponent = pow(meas_x-mu_x,2)/(2*pow(std_landmark[0],2))+pow(meas_y-mu_y,2)/(2*pow(std_landmark[1],2));
                    long double multipler = gauss_norm*exp(-exponent);
                    if(multipler>0)
                    {
                        particles[p].weight*=multipler;
                    }
                }

                sense_x.push_back(trans_observations[j].x);
                sense_y.push_back(trans_observations[j].y);
            }

            weights[p]=particles[p].weight;
        }
}


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(),weights.end());
    vector<Particle> temp_pas;
    for (unsigned int i=0;i<num_particles;i++){
        temp_pas.push_back(particles[distribution(gen)]);
    }
    particles = temp_pas;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
