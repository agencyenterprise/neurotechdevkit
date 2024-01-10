import { createStore } from "vuex";

const store = createStore({
  state() {
    return {
      is2d: true,
      isPreBuilt: true,
      hasSimulation: false,
      isRunningSimulation: false,
      ctScan: null,
      scenario: null,
      configuration: {},
      builtInScenarios: [],
      materials: [],
      materialProperties: [],
      transducerTypes: [],
      availableCTs: [],
    };
  },
  mutations: {
    set2d(state, payload) {
      state.is2d = payload;
    },
    setIsPreBuilt(state, payload) {
      state.isPreBuilt = payload;
    },
    setInitialValues(state, payload) {
      console.log("initial values will be:", payload);
      state.hasSimulation = payload.has_simulation;
      state.isRunningSimulation = payload.is_running_simulation;
      state.configuration = payload.configuration;
      state.builtInScenarios = payload.built_in_scenarios;
      state.materials = payload.materials;
      state.materialProperties = payload.material_properties;
      state.transducerTypes = payload.transducer_types;
      state.availableCTs = payload.available_cts;
    },
    setScenario(state, payload) {
      state.scenario = payload;
    },
    setCTScan(state, payload) {
      state.ctScan = payload;
    },
  },
  actions: {
    set2d(context, payload) {
      context.commit("set2d", payload);
    },
    setIsPreBuilt(context, payload) {
      context.commit("setIsPreBuilt", payload);
    },
    setInitialValues(context, payload) {
      context.commit("setInitialValues", payload);
    },
    setScenario(context, payload) {
      // payload will have the scenario name, we need to iterate over the
      // builtInScenarios to find the scenario and then set the scenario in the state
      const scenarios = context.getters.builtInScenarios;
      const scenario = scenarios[payload];
      context.commit("setScenario", scenario);
    },
    setCTScan(context, payload) {
      // payload will be the ct scan filename, we need to iterate over the
      // availableCTs to find the ct scan and then set the ct scan in the state
      const ctScans = context.getters.ctScans;
      const ctScan = ctScans.filter((ct) => ct.filename === payload)[0];
      context.commit("setCTScan", ctScan);
    },
  },
  getters: {
    is2d(state) {
      return state.is2d;
    },
    isPreBuilt(state) {
      return state.isPreBuilt;
    },
    ctScans(state) {
      return state.availableCTs;
    },
    builtInScenarios(state) {
      return state.builtInScenarios;
    },
    scenario(state) {
      return state.scenario;
    },
    builtInScenarios2d(state) {
      const scenarios = state.builtInScenarios;
      const filteredScenarios = Object.keys(scenarios)
        .filter((key) => scenarios[key].is2d)
        .reduce((obj, key) => {
          obj[key] = scenarios[key];
          return obj;
        }, {});
      return filteredScenarios;
    },
    builtInScenarios3d(state) {
      const scenarios = state.builtInScenarios;
      const filteredScenarios = Object.keys(scenarios)
        .filter((key) => !scenarios[key].is2d)
        .reduce((obj, key) => {
          obj[key] = scenarios[key];
          return obj;
        }, {});
      return filteredScenarios;
    },
  },
});

export default store;
