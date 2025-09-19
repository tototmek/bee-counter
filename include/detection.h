#ifndef DETECTION_H
#define DETECTION_H

#include <cstddef>
#include <set>

constexpr int kFilterWindow = 15;
constexpr int kDetrendWindow = 850;
constexpr float kThreshold = 215.0f;

constexpr int kKernelLength = 141;
const float kernel[kKernelLength] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0,-0.08351139097398678,-0.16669320064052467,-0.24921714839901377,-0.3307575499292569,-0.4109926025186801,-0.48960565507062176,-0.5662864577815467,-0.6407323865552814,-0.7126496373220655,-0.7817543855489894,-0.8477739063657574,-0.910447650885156,-0.9695282744704774,-1.0247826128917998,-1.0759926025186801,-1.12295614091768,-1.1654878844583387,-1.2034199797798062,-1.2366027262313746,-1.2649051666725544,-1.2882156043010795,-1.3064420434691562,-1.3195125527482556,-1.3273755488096013,-1.33,-1.3273755488096013,-1.3195125527482554,-1.306442043469156,-1.2882156043010795,-1.2649051666725544,-1.2366027262313743,-1.203419979779806,-1.1654878844583385,-1.12295614091768,-1.0759926025186801,-1.0247826128917998,-0.9695282744704773,-0.9104476508851558,-0.8477739063657571,-0.7817543855489895,-0.7126496373220655,-0.6407323865552813,-0.5662864577815464,-0.4896056550706214,-0.4109926025186797,-0.33075754992925693,-0.24921714839901368,-0.16669320064052445,-0.08351139097398648,4.2776062481398537e-16,0.08351139097398676,0.1666932006405247,0.24921714839901396,0.3307575499292572,0.4109926025186805,0.4896056550706222,0.5662864577815466,0.6407323865552815,0.7126496373220658,0.7817543855489896,0.8477739063657577,0.910447650885156,0.9695282744704774,1.0247826128917998,1.0759926025186801,1.1229561409176805,1.1654878844583387,1.2034199797798064,1.2366027262313746,1.2649051666725544,1.2882156043010795,1.3064420434691562,1.3195125527482556,1.3273755488096013,1.33,1.3273755488096013,1.3195125527482556,1.306442043469156,1.2882156043010795,1.2649051666725544,1.2366027262313741,1.2034199797798062,1.1654878844583383,1.12295614091768,1.0759926025186797,1.0247826128917994,0.9695282744704774,0.9104476508851554,0.8477739063657572,0.7817543855489886,0.7126496373220651,0.6407323865552814,0.5662864577815461,0.4896056550706216,0.41099260251867925,0.3307575499292565,0.24921714839901385,0.16669320064052404,0.08351139097398665,3.2575604857319597e-16,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}; 

template<size_t N>
class CircularBuffer {
  public:
    CircularBuffer() : m_head(0) {
        for (size_t i = 0; i < N; ++i) {
            m_buffer[i] = 0.0f;
        }
    }

    void push_back(float value) {
        m_buffer[m_head] = value;
        m_head = (m_head + 1) % N;
    }

    float at(size_t index) const {
        if (index >= N) {
            return 0.0f; 
        }
        size_t physicalIndex = (m_head + index) % N;
        return m_buffer[physicalIndex];
    }
    
    size_t size() const {
        return N;
    }

  private:
    float m_buffer[N];
    size_t m_head;
};

class RollingAverageFilter {
  public:
    float update(float newValue) {
        // More efficient implementation of online averaging filter
        average -= x.at(0) / float(kFilterWindow);
        x.push_back(newValue);
        average += newValue / float(kFilterWindow);
        return average;
    }

  private:
    CircularBuffer<kFilterWindow> x;
    float average = 0;
};


class RollingMedianDetrender {
public:
    // Efficient O(log(n)) implementation
    RollingMedianDetrender() {
        // Initialize the buffer with zeros and populate the multisets.
        for (size_t i = 0; i < kDetrendWindow; ++i) {
            float value = x.at(i);
            lower_half.insert(value);
            rebalance();
        }
    }

    float update(float newValue) {
        // Get the oldest value to remove from the filter window.
        float oldestValue = x.at(0);

        x.push_back(newValue);

        // Remove the oldest value from the correct multiset.
        if (oldestValue <= *lower_half.rbegin()) {
            lower_half.erase(lower_half.find(oldestValue));
        } else {
            upper_half.erase(upper_half.find(oldestValue));
        }

        // Insert the new value into the correct multiset.
        if (!lower_half.empty() && newValue > *lower_half.rbegin()) {
            upper_half.insert(newValue);
        } else {
            lower_half.insert(newValue);
        }

        rebalance();

        return newValue - *lower_half.rbegin();
    }

private:
    void rebalance() {
        // Ensure that lower_half has one more element than upper_half.
        while (lower_half.size() > upper_half.size() + 1) {
            // Move an element from lower_half to upper_half.
            upper_half.insert(*lower_half.rbegin());
            lower_half.erase(std::prev(lower_half.end()));
        }
        // Ensure that upper_half doesn't have more elements than lower_half.
        while (upper_half.size() > lower_half.size()) {
            // Move an element from upper_half to lower_half.
            lower_half.insert(*upper_half.begin());
            upper_half.erase(upper_half.begin());
        }
    }
    CircularBuffer<kDetrendWindow> x;
    std::multiset<float> lower_half;
    std::multiset<float> upper_half;
};


class Correlator {
  public:
    float update(float newValue) {
        x.push_back(newValue);

        // Compute cross-correlation with signal model
        float R = 0;
        for (int tau = 0; tau < kKernelLength; ++tau) {
            R += kernel[tau] * x.at(tau);
        }

        return R;
    }

  private:
    CircularBuffer<kKernelLength> x;
};


class Detector {
  public:
    int update(float newValue) { // returns 1 if a bee entered, -1 if a bee leaved, and 0 otherwise.
        float signal = newValue;
        signal = filter.update(signal); 
        signal = detrender.update(signal);
        signal = correlator.update(signal);

        int state = 0;
        if (signal < -kThreshold) { state = 1; }
        else if (signal > kThreshold) { state = -1; }

        int output = 0;
        if (prevState != 1 and state == 1) {
            output = 1;
        } else if (prevState != -1 and state == -1) {
            output = -1;
        }

        prevState = state;
        return output;
    }
    
  private:
    RollingAverageFilter filter;
    RollingMedianDetrender detrender;
    Correlator correlator;
    int prevState = 0;

};



#endif // DETECTION H