

template <typename T>
struct AdvectionFirstGuess
{
    __host__ __device__
    float operator()(const T& p, const T& vInit, const T& dt) const {
        // new_position = cur_postion + velocity(t) * dt
        return p + vInit * dt;
    }
};


template <typename T>
struct AdvectionFinalPosition
{
    __host__ __device__
    float operator()(const T& p, const T& vInit, const T& vGuess, const T& dt) const {
        // use average of initial velocity and guess velocity
        //   new_position = cur_postion + 0.5 * (vInit + vGuess) * dt
        return p + 0.5 * (vInit + vGuess) * dt;
    }
};

