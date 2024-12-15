const { Sequelize } = require('sequelize');
require('dotenv').config({ path: './.env' })

console.log('Database URL:', process.env.DATABASE_URL);

const sequelize = new Sequelize(process.env.DATABASE_URL, {
  dialect: 'mysql',
  logging: false,
});

sequelize
  .authenticate()
  .then(() => {
    console.log('Connection to the database has been established successfully.');
  })
  .catch(err => {
    console.error('Unable to connect to the database:', err);
  });
  const mysql = require('mysql2');

  const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'dheerizz', // Replace with your actual password
    database: 'gatewebsite'
  });
  
  db.connect((err) => {
    if (err) {
      console.error('Error connecting to database:', err);
    } else {
      console.log('Connected to database');
    }
  });
  
  module.exports = db;
module.exports = sequelize;
