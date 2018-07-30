import React, { Component } from "react";
import "./App.css";

import { withStyles } from "@material-ui/core/styles";

import Typography from "@material-ui/core/Typography";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import Select from "@material-ui/core/Select";
import Button from '@material-ui/core/Button';
import CircularProgress from '@material-ui/core/CircularProgress';

import axios from 'axios';


class App extends Component {
  state = {
    loading: false,
    x: "",
    y: "",
    model: "",
    result: ""
  };

  handleChange = event => {
    this.setState({ [event.target.name]: event.target.value });
  };

  submit = () => {
    this.setState({ loading: true });
    const { x, y, model } = this.state;
    console.log("request:", x, y, model)
    axios
      .get("http://localhost:8080/", {
        params: {
          x: x,
          y: y,
          model: model
        }
      })
      .then(res => {
        console.log("response:", x, y, model)
        console.log("result:", res.data[0])

        this.setState({ result: res.data[0], loading: false });
      });
  }


  render() {
    const { result, loading } = this.state;

    return (

      <div className="App">


        <br />
        <br />
        <form autoComplete="off">
          <FormControl>
            <InputLabel htmlFor="controlled-open-select">X1</InputLabel>
            <br />
            <br />
            <Select
              open={this.state.open}
              onClose={this.handleClose}
              onOpen={this.handleOpen}
              value={this.state.x}
              onChange={this.handleChange}
              inputProps={{
                name: "x",
                id: "controlled-open-select-x"
              }}
            >
              <MenuItem value={0}>0</MenuItem>
              <MenuItem value={1}>1</MenuItem>
            </Select>
          </FormControl>{" "}
          <FormControl>
            <InputLabel htmlFor="controlled-open-select">X2</InputLabel>
            <br />
            <br />
            <Select
              open={this.state.open}
              onClose={this.handleClose}
              onOpen={this.handleOpen}
              value={this.state.y}
              onChange={this.handleChange}
              inputProps={{
                name: "y",
                id: "controlled-open-select-y"
              }}
            >
              <MenuItem value={0}>0</MenuItem>
              <MenuItem value={1}>1</MenuItem>
            </Select>
          </FormControl>{" "}
          <FormControl>
            <InputLabel htmlFor="controlled-open-select">Modèle</InputLabel>
            <br />
            <br />
            <Select
              open={this.state.open}
              onClose={this.handleClose}
              onOpen={this.handleOpen}
              value={this.state.model}
              onChange={this.handleChange}
              inputProps={{
                name: "model",
                id: "controlled-open-select-model"
              }}
            >
              <MenuItem value={'or'}>OR</MenuItem>
              <MenuItem value={'and'}>AND</MenuItem>
              <MenuItem value={'xor'}>XOR</MenuItem>
              <MenuItem value={'xnor'}>XNOR</MenuItem>
            </Select>
          </FormControl>
        </form>

        <br />
        <br />
        <Button variant="raised" color="secondary" onClick={this.submit}>
          Clique Moi
      </Button>
        <br />
        <br />

        <Typography variant="display1" gutterBottom>
          Réponse: {loading ? (<CircularProgress />) : result}
        </Typography>

      </div>
    );
  }
}

const styles = theme => ({});

export default withStyles(styles)(App);
