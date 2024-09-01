




class ConfigParam :

    TRANSLATIONAL_STIFFNESS = {
        "level" : 0,
        "decription" :"Cartesian translational stiffness",
        "default" : 2000,
        "min" : 0,
        "max" : 6000
    }

    TRANSLATION_DAMPING = {
        "level" : 0,
        "decription" :"Cartesian translational damping",
        "default" : 89,
        "min" : 0,
        "max" : 400
    }

    ROTATIONAL_STIFFNESS = {
        "level" : 0,
        "decription" :"Cartesian rotational stiffness",
        "default" : 150,
        "min" : 0,
        "max" : 300
    }

    ROTATIONAL_DAMPING = {
        "level" : 0,
        "decription" :"Cartesian rotational damping",
        "default" : 7,
        "min" : 0,
        "max" : 30
    }
    NULLSPACE_STIFFNESS = {
        "level" : 0,
        "decription" :"Stiffness of the joint space nullspace controller (the desired configuration is the one at startup)",
        "default" : 0.2,
        "min" : 0,
        "max" : 100
    }
    JOINT1_NULLSPACE_STIFFNESS = {
        "level" : 0,
        "decription" :"Stiffness of the joint space nullspace controller (the desired configuration is the one at startup)",
        "default" : 100,
        "min" : 0,
        "max" : 100
    }
    TRANSLATIONAL_CLIP_NEG_X = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_CLIP_NEG_Y = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_CLIP_NEG_Z = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_CLIP_X = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_CLIP_Y = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_CLIP_Z = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.01,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_NEG_X = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_NEG_Y = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_NEG_Z = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_X = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_Y = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    ROTATIONAL_CLIP_Z = {
        "level" : 0,
        "decription" :"Value to clip error to in realtime control loop",
        "default" : 0.05,
        "min" : 0,
        "max" : 0.1
    }
    TRANSLATIONAL_KI = {
        "level" : 0,
        "decription" :"Cartesian translational intergral gain",
        "default" : 0,
        "min" : 0,
        "max" : 100
    }
    ROTATIONAL_KI = {
        "level" : 0,
        "decription" :"Cartesian rotational intergral gain",
        "default" : 0,
        "min" : 0,
        "max" : 100
    }